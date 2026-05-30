"""
RTMOPose: Real-time multi-person pose estimation.
Backbone: HGNetV2 (same as DETRPose)
Neck:     PAFPN
Head:     Decoupled cls/bbox/kpt (anchor-free, FCOS-style)
Matcher:  SimOTA (GPU-native)
"""

import torch
import torch.nn as nn

from .neck import PAFPN
from .head import RTMOHead
from .postprocesses import RTMOPostProcess


class RTMOPose(nn.Module):
    """
    RTMOPose single-stage pose estimation model.

    Args:
        backbone:          nn.Module – HGNetV2 backbone (instantiated externally via config).
        neck_out_channels: Unified PAFPN output channels (256).
        neck_depth_mult:   Depth multiplier for PAFPN CSP blocks.
        num_classes:       Number of object classes.
        num_body_points:   Keypoints per instance.
        feat_strides:      Strides corresponding to FPN levels, e.g. [8, 16, 32].
        head_num_convs:    Conv depth per head branch.
        act:               Activation name ('silu', 'relu', …).
        post_processor:    Optional RTMOPostProcess instance (used in eval/export).
    """

    def __init__(
        self,
        backbone,
        neck_out_channels=256,
        neck_depth_mult=1.0,
        num_classes=80,
        num_body_points=17,
        feat_strides=(8, 16, 32),
        head_num_convs=2,
        act="silu",
        post_processor=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_body_points = num_body_points

        # Backbone output channels come from backbone.num_channels
        # (HGNetV2 returns len(return_idx) feature levels)
        backbone_channels = backbone.num_channels   # e.g. [256, 512, 1024] for B0

        self.neck = PAFPN(
            in_channels=backbone_channels,
            out_channels=neck_out_channels,
            depth_mult=neck_depth_mult,
            act=act,
        )

        self.head = RTMOHead(
            in_channels=neck_out_channels,
            num_classes=num_classes,
            num_body_points=num_body_points,
            feat_strides=feat_strides,
            num_convs=head_num_convs,
            act=act,
        )

        self.post_processor = post_processor

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, samples, targets=None):
        """
        Args:
            samples:  NestedTensor or plain [B, 3, H, W] tensor.
            targets:  list of target dicts (used in training to pass to criterion).
                      Not consumed here – the criterion is called from the solver.
        Returns:
            During training:  dict with raw head outputs (passed to RTMOCriterion)
            During eval:      list of per-image result dicts (post-processed)
        """
        # Handle NestedTensor (same pattern as DETRPose)
        if hasattr(samples, "tensors"):
            x = samples.tensors
        else:
            x = samples

        # Backbone
        feats = self.backbone(x)   # list of feature maps [stage2, stage3, stage4]

        # Neck
        neck_feats = self.neck(feats)

        # Head
        cls_logits, bbox_pred, kpt_pred, anchor_pts, strides = self.head(neck_feats)

        # Decode predictions for loss / postprocess
        # Do this once here so both criterion and postprocessor can share them
        bbox_decoded = self._decode_bbox_all(bbox_pred, anchor_pts, strides)
        kpts_xy, kpts_vis = self.head.decode_keypoints(
            kpt_pred.flatten(0, 1),
            bbox_decoded.flatten(0, 1),
        )
        B = cls_logits.shape[0]
        N = cls_logits.shape[1]
        K = self.num_body_points
        kpts_xy  = kpts_xy.view(B, N, K, 2)
        kpts_vis = kpts_vis.view(B, N, K)

        # raw visibility logits for BCE loss
        kpt_vis_raw = kpt_pred[..., K * 2:]  # [B, N, K]

        outputs = {
            "cls_logits":   cls_logits,    # [B, N, C]
            "bbox_pred":    bbox_pred,      # [B, N, 4]  raw ltrb
            "kpt_pred":     kpt_pred,       # [B, N, K*3]
            "anchor_pts":   anchor_pts,     # [N, 2]
            "strides":      strides,        # [N]
            "bbox_decoded": bbox_decoded,   # [B, N, 4]  xyxy absolute
            "kpts_decoded": kpts_xy,        # [B, N, K, 2]
            "kpts_vis":     kpts_vis,       # [B, N, K]  sigmoid scores
            "kpt_vis_raw":  kpt_vis_raw,    # [B, N, K]  raw logits
        }

        if not self.training and self.post_processor is not None:
            # Build target sizes for clamping: use the input image size
            B_, _, H_, W_ = x.shape
            orig_sizes = x.new_tensor([[H_, W_]] * B_)
            return self.post_processor(outputs, orig_sizes)

        return outputs

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _decode_bbox_all(bbox_pred, anchor_pts, strides):
        """
        Decode [B, N, 4] ltrb → [B, N, 4] xyxy absolute.
        Uses RTMOHead.decode_bbox per image.
        """
        B, N, _ = bbox_pred.shape
        bbox_flat = bbox_pred.flatten(0, 1)           # [B*N, 4]
        ap = anchor_pts.unsqueeze(0).expand(B, -1, -1).flatten(0, 1)  # [B*N, 2]
        st = strides.unsqueeze(0).expand(B, -1).flatten()              # [B*N]
        xyxy = RTMOHead.decode_bbox(bbox_flat, ap, st)
        return xyxy.view(B, N, 4)
