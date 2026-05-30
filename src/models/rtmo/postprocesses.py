"""
RTMOPose PostProcessor.
Converts raw model outputs → same format as DETRPose PostProcess:
  [{'scores': [N], 'labels': [N], 'boxes': [N, 4], 'keypoints': [N, K*3]}]
Boxes are x1y1x2y2 absolute pixels.
Keypoints are [x1, y1, vis1, x2, y2, vis2, ...] absolute pixels.
"""

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from .head import RTMOHead


class RTMOPostProcess(nn.Module):
    """
    Args:
        score_threshold:  Pre-NMS confidence filter.
        nms_threshold:    IoU threshold for NMS.
        max_detections:   Keep at most this many detections per image.
        deploy_mode:      If True return raw tensors (for TensorRT export).
    """

    def __init__(
        self,
        score_threshold=0.3,
        nms_threshold=0.65,
        max_detections=300,
        deploy_mode=False,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.deploy_mode = deploy_mode

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes):
        """
        Args:
            outputs: dict from RTMOPose.forward() (training mode) with keys:
                'cls_logits'   [B, N, C]
                'bbox_decoded' [B, N, 4]  xyxy absolute (in net input space)
                'kpts_decoded' [B, N, K, 2] absolute
                'kpts_vis'     [B, N, K]  vis scores (sigmoid)
                'anchor_pts'   [N, 2]
                'strides'      [N]
                OR a list of result dicts (when called after model.eval() inline).
            orig_target_sizes: [B, 2] tensor of (H, W) for each image
                               (used to clamp / scale back if the model
                                was given a resized image — pass the net
                                input size if you don't want scaling).
        Returns:
            list of dicts, one per image.
        """
        # When model is in eval mode it returns results list directly;
        # this postprocessor may be called as a passthrough from the engine.
        if isinstance(outputs, list):
            return outputs

        cls_logits   = outputs["cls_logits"]    # [B, N, C]
        bbox_decoded = outputs["bbox_decoded"]   # [B, N, 4]
        kpts_decoded = outputs["kpts_decoded"]   # [B, N, K, 2]
        kpts_vis     = outputs["kpts_vis"]       # [B, N, K]

        B = cls_logits.shape[0]
        scores_all = cls_logits.sigmoid()  # [B, N, C]

        results = []
        for b in range(B):
            scores = scores_all[b]    # [N, C]
            boxes  = bbox_decoded[b]  # [N, 4]
            kpts   = kpts_decoded[b]  # [N, K, 2]
            vis    = kpts_vis[b]      # [N, K]

            # Best score and class per anchor
            max_scores, labels = scores.max(-1)  # [N], [N]

            # Pre-NMS score threshold
            keep = max_scores >= self.score_threshold
            if keep.sum() == 0:
                results.append({
                    "scores":    torch.zeros(0, device=cls_logits.device),
                    "labels":    torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    "boxes":     torch.zeros(0, 4, device=cls_logits.device),
                    "keypoints": torch.zeros(0, kpts.shape[1] * 3, device=cls_logits.device),
                })
                continue

            max_scores = max_scores[keep]
            labels     = labels[keep]
            boxes      = boxes[keep]
            kpts       = kpts[keep]
            vis        = vis[keep]

            # NMS (class-aware via offset)
            keep_nms = batched_nms(boxes, max_scores, labels, self.nms_threshold)
            if keep_nms.shape[0] > self.max_detections:
                keep_nms = keep_nms[:self.max_detections]

            max_scores = max_scores[keep_nms]
            labels     = labels[keep_nms]
            boxes      = boxes[keep_nms]
            kpts       = kpts[keep_nms]      # [M, K, 2]
            vis        = vis[keep_nms]       # [M, K]

            # Clamp boxes to image
            if orig_target_sizes is not None:
                h, w = orig_target_sizes[b, 0].item(), orig_target_sizes[b, 1].item()
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

            # Build keypoints tensor: [M, K*3] → [x, y, vis, x, y, vis, ...]
            M, K, _ = kpts.shape
            kpts_flat = torch.cat([kpts, vis.unsqueeze(-1)], dim=-1)  # [M, K, 3]
            kpts_flat = kpts_flat.reshape(M, K * 3)

            if self.deploy_mode:
                results.append({
                    "scores":    max_scores,
                    "labels":    labels,
                    "boxes":     boxes,
                    "keypoints": kpts_flat,
                })
            else:
                results.append({
                    "scores":    max_scores,
                    "labels":    labels,
                    "boxes":     boxes,
                    "keypoints": kpts_flat,
                })

        return results
