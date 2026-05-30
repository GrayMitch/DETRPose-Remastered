"""
RTMOPose Head: Decoupled classification / bbox / keypoint prediction head.
Point-based anchor-free (FCOS-style ltrb box encoding).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detrpose.hybrid_encoder import get_activation, ConvNormLayer


class DWConv(nn.Module):
    """Depth-wise separable conv + BN + activation."""

    def __init__(self, channels, kernel_size=3, act="silu"):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv2d(
            channels, channels, kernel_size,
            padding=padding, groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


def _make_branch(in_ch, num_convs, act):
    layers = [DWConv(in_ch, act=act) for _ in range(num_convs)]
    return nn.Sequential(*layers)


class RTMOHead(nn.Module):
    """
    Decoupled single-stage head for class + bbox + keypoints.

    Bbox encoding: ltrb distances (l/t/r/b from anchor point to box edge),
                   divided by stride so values are ~O(1).
    Keypoints:     (num_kpts * 2) xy offsets relative to bbox center,
                   normalised by stride + (num_kpts) visibility logits.

    Args:
        in_channels:     Feature channels from neck.
        num_classes:     Number of object classes.
        num_body_points: Keypoints per instance.
        feat_strides:    Strides of each neck level, e.g. [8, 16, 32].
        num_convs:       Depth of each branch.
        act:             Activation name.
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=80,
        num_body_points=17,
        feat_strides=(8, 16, 32),
        num_convs=2,
        act="silu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.feat_strides = list(feat_strides)
        num_levels = len(feat_strides)

        # Per-level branches (shared weights would also be fine)
        self.cls_branches  = nn.ModuleList([_make_branch(in_channels, num_convs, act) for _ in range(num_levels)])
        self.bbox_branches = nn.ModuleList([_make_branch(in_channels, num_convs, act) for _ in range(num_levels)])
        self.kpt_branches  = nn.ModuleList([_make_branch(in_channels, num_convs, act) for _ in range(num_levels)])

        # Prediction heads
        self.cls_preds  = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in range(num_levels)])
        self.bbox_preds = nn.ModuleList([nn.Conv2d(in_channels, 4, 1) for _ in range(num_levels)])          # ltrb
        self.kpt_preds  = nn.ModuleList([nn.Conv2d(in_channels, num_body_points * 3, 1) for _ in range(num_levels)])  # xy + vis per kpt

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        prior_prob = 0.01
        bias_cls = -math.log((1 - prior_prob) / prior_prob)
        for pred in self.cls_preds:
            nn.init.constant_(pred.bias, bias_cls)
            nn.init.normal_(pred.weight, std=0.01)
        for pred in self.bbox_preds:
            nn.init.constant_(pred.bias, 1.0)   # ~1 stride unit from center
            nn.init.normal_(pred.weight, std=0.01)
        for pred in self.kpt_preds:
            nn.init.constant_(pred.bias, 0.0)
            nn.init.normal_(pred.weight, std=0.01)

    # ------------------------------------------------------------------
    def forward(self, feats):
        """
        Args:
            feats: list of [B, C, Hi, Wi]  (one per FPN level)
        Returns:
            cls_preds_all:  [B, total_anchors, num_classes]  – raw logits
            bbox_preds_all: [B, total_anchors, 4]            – ltrb in stride units (raw, before softplus)
            kpt_preds_all:  [B, total_anchors, num_kpts*3]   – xy_offset (normalised) + vis logits
            anchor_points:  [total_anchors, 2]               – absolute xy of each anchor point
            strides_all:    [total_anchors]                  – stride of each anchor point
        """
        cls_list, bbox_list, kpt_list = [], [], []
        anchor_pts_list, stride_list = [], []

        for i, (feat, stride) in enumerate(zip(feats, self.feat_strides)):
            B, C, H, W = feat.shape

            # Anchor grid (centre of each cell, absolute pixels)
            ys, xs = torch.meshgrid(
                torch.arange(H, device=feat.device, dtype=feat.dtype) + 0.5,
                torch.arange(W, device=feat.device, dtype=feat.dtype) + 0.5,
                indexing="ij"
            )
            anchors = torch.stack([xs, ys], dim=-1).reshape(-1, 2) * stride  # [H*W, 2]
            anchor_pts_list.append(anchors)
            stride_list.append(torch.full((H * W,), stride, device=feat.device, dtype=feat.dtype))

            # Predictions
            cls  = self.cls_preds[i](self.cls_branches[i](feat))    # [B, num_cls, H, W]
            bbox = self.bbox_preds[i](self.bbox_branches[i](feat))  # [B, 4,       H, W]
            kpt  = self.kpt_preds[i](self.kpt_branches[i](feat))    # [B, K*3,     H, W]

            cls_list.append( cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            bbox_list.append(bbox.permute(0, 2, 3, 1).reshape(B, -1, 4))
            kpt_list.append( kpt.permute(0, 2, 3, 1).reshape(B, -1, self.num_body_points * 3))

        cls_preds_all  = torch.cat(cls_list,  dim=1)
        bbox_preds_all = torch.cat(bbox_list, dim=1)
        kpt_preds_all  = torch.cat(kpt_list,  dim=1)
        anchor_points  = torch.cat(anchor_pts_list, dim=0)
        strides_all    = torch.cat(stride_list, dim=0)

        return cls_preds_all, bbox_preds_all, kpt_preds_all, anchor_points, strides_all

    # ------------------------------------------------------------------
    @staticmethod
    def decode_bbox(bbox_pred, anchor_points, strides):
        """
        Decode ltrb predictions → xyxy absolute coords.

        Args:
            bbox_pred:     [N, 4]  – raw ltrb logits
            anchor_points: [N, 2]  – absolute xy anchor centres
            strides:       [N]     – stride per anchor
        Returns:
            boxes_xyxy: [N, 4]  – x1y1x2y2 in absolute pixels
        """
        dist = F.softplus(bbox_pred) * strides[:, None]   # ensure positive, scale
        x1 = anchor_points[:, 0] - dist[:, 0]
        y1 = anchor_points[:, 1] - dist[:, 1]
        x2 = anchor_points[:, 0] + dist[:, 2]
        y2 = anchor_points[:, 1] + dist[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def decode_keypoints(kpt_pred, boxes_xyxy):
        """
        Decode keypoint predictions → absolute xy + visibility.

        Offsets are predicted relative to bbox center, normalised by
        bbox half-dimensions.

        Args:
            kpt_pred:   [N, num_kpts*3]  – (dx, dy, vis) per keypoint (raw)
            boxes_xyxy: [N, 4]
        Returns:
            kpts_xy:  [N, num_kpts, 2]  – absolute pixel coordinates
            kpts_vis: [N, num_kpts]     – visibility scores (sigmoid)
        """
        num_kpts = kpt_pred.shape[-1] // 3
        xy_offset = kpt_pred[..., :num_kpts * 2].reshape(-1, num_kpts, 2)
        vis_logit = kpt_pred[..., num_kpts * 2:]

        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
        hw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1)
        hh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1)

        kpts_xy = torch.stack([
            cx[:, None] + xy_offset[..., 0] * hw[:, None],
            cy[:, None] + xy_offset[..., 1] * hh[:, None],
        ], dim=-1)

        return kpts_xy, vis_logit.sigmoid()
