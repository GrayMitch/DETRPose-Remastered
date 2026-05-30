"""
RTMOPose Loss / Criterion.
Losses: VariFocal (classification), CIoU (bbox), OKS (keypoints), BCE (visibility).
Assignment: SimOTA (see assigner.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import simota_assign
from .head import RTMOHead
from src.misc.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def varifocal_loss(pred_logits, target_score, target_label, alpha=0.75, gamma=2.0):
    """
    Varifocal loss (Zhang et al. 2021).
    pred_logits:  [N, C]  raw
    target_score: [N, C]  soft labels (IoU-weighted for positives, 0 for negatives)
    Returns scalar.
    """
    pred_prob = pred_logits.sigmoid()
    weight = alpha * (pred_prob - target_score).abs().pow(gamma) * (target_score > 0).float() + \
             (1 - alpha) * pred_prob.pow(gamma) * (target_score == 0).float()
    loss = F.binary_cross_entropy_with_logits(pred_logits, target_score, reduction="none")
    return (loss * weight).sum()


def ciou_loss(pred_xyxy, gt_xyxy, eps=1e-7):
    """
    Complete IoU loss between two sets of aligned boxes.
    pred_xyxy, gt_xyxy: [N, 4] in x1y1x2y2 format.
    Returns [N] losses.
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    # Intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    # Enclosing box
    ex1 = torch.min(px1, gx1)
    ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2)
    ey2 = torch.max(py2, gy2)
    c2 = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    # Centre distance²
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    rho2 = (pcx - gcx).pow(2) + (pcy - gcy).pow(2)

    # Aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)).pow(2)
    with torch.no_grad():
        alpha_v = v / (1 - iou + v + eps)

    return 1.0 - iou + rho2 / c2 + alpha_v * v  # [N]


def oks_loss_rtmo(kpts_xy, gt_kpts_xy, gt_vis, gt_area, sigmas, eps=1e-6):
    """
    OKS loss: -log(OKS).

    kpts_xy:    [N, K, 2]  predicted absolute coords
    gt_kpts_xy: [N, K, 2]  gt absolute coords
    gt_vis:     [N, K]     visibility (0/1/2 → treated as valid if > 0)
    gt_area:    [N]        object area (used as normaliser)
    sigmas:     [K]        per-keypoint sigma
    """
    variances = (sigmas * 2) ** 2   # [K]
    variances = variances.to(gt_area.device)
    valid = (gt_vis > 0).float()    # [N, K]

    d2 = ((kpts_xy - gt_kpts_xy) ** 2).sum(-1)  # [N, K]
    s2 = gt_area[:, None] * variances[None, :]   # [N, K]  (broadcast)
    oks = (torch.exp(-d2 / (2 * s2 + eps)) * valid).sum(-1) / (valid.sum(-1) + eps)
    return -oks.clamp(min=eps).log()  # [N]


# ─────────────────────────────────────────────────────────────────────────────
# Main criterion
# ─────────────────────────────────────────────────────────────────────────────

class RTMOCriterion(nn.Module):
    """
    Computes the training losses for RTMOPose.

    Args:
        num_classes:       Number of object categories.
        num_body_points:   Keypoints per instance.
        sigmas:            Per-keypoint sigma array (length = num_body_points).
        weight_dict:       Loss weight multipliers, keys:
                             'loss_cls', 'loss_bbox', 'loss_kpts', 'loss_vis'
        topk_candidates:   SimOTA top-k for candidate selection.
    """

    def __init__(
        self,
        num_classes,
        num_body_points,
        sigmas,
        weight_dict=None,
        topk_candidates=13,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_body_points = num_body_points
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.topk = topk_candidates
        self.weight_dict = weight_dict or {
            "loss_cls": 1.0,
            "loss_bbox": 2.0,
            "loss_kpts": 5.0,
            "loss_vis": 1.0,
        }

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys:
                'cls_logits'   [B, N, C]
                'bbox_pred'    [B, N, 4]   raw ltrb
                'kpt_pred'     [B, N, K*3]
                'anchor_pts'   [N, 2]
                'strides'      [N]
                'bbox_decoded' [B, N, 4]   decoded xyxy absolute
                'kpts_decoded' [B, N, K, 2] decoded absolute
                'kpts_vis'     [B, N, K]    vis scores (sigmoid already applied in head decode? no, use raw for BCE)
            targets: list of dicts (one per image) with:
                'boxes'     [G, 4]   xyxy absolute
                'labels'    [G]
                'keypoints' [G, K, 3]  (x, y, vis) where vis=0/1/2
                'area'      [G]
        Returns:
            loss_dict: {'loss_cls', 'loss_bbox', 'loss_kpts', 'loss_vis'}
        """
        cls_logits   = outputs["cls_logits"]    # [B, N, C]
        bbox_decoded = outputs["bbox_decoded"]   # [B, N, 4] xyxy
        kpts_decoded = outputs["kpts_decoded"]   # [B, N, K, 2]
        kpt_vis_raw  = outputs["kpt_vis_raw"]    # [B, N, K]  raw logits for BCE
        anchor_pts   = outputs["anchor_pts"]     # [N, 2]
        strides      = outputs["strides"]        # [N]

        B, N, C = cls_logits.shape
        device = cls_logits.device

        total_cls  = cls_logits.new_zeros(())
        total_bbox = cls_logits.new_zeros(())
        total_kpts = cls_logits.new_zeros(())
        total_vis  = cls_logits.new_zeros(())
        num_fg = 0

        for b_idx in range(B):
            tgt = targets[b_idx]
            gt_boxes  = tgt["boxes"].to(device)      # [G, 4] normalized cxcywh (from Normalize transform)
            gt_labels = tgt["labels"].to(device)     # [G]
            gt_kpts   = tgt["keypoints"].to(device)  # [G, K*3] flat normalized (from Normalize transform)
            gt_area   = tgt.get("area", None)

            # Un-normalize targets — Normalize transform outputs normalized coords
            img_size = tgt["size"]
            H_img = float(img_size[0])
            W_img = float(img_size[1])
            scale_box = gt_boxes.new_tensor([W_img, H_img, W_img, H_img])
            gt_boxes = box_cxcywh_to_xyxy(gt_boxes * scale_box)
            K = self.num_body_points
            gt_kpts_xy  = gt_kpts[:, :K * 2].view(-1, K, 2) * gt_kpts.new_tensor([W_img, H_img])
            gt_kpts_vis = gt_kpts[:, K * 2:]                  # [G, K]
            gt_kpts = torch.cat([gt_kpts_xy, gt_kpts_vis.unsqueeze(-1)], dim=-1)  # [G, K, 3]

            # Compute area from boxes if not provided, else scale from normalized
            if gt_area is None:
                bw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
                bh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
                gt_area = bw * bh
            else:
                gt_area = gt_area.to(device) * (H_img * W_img)

            # Image size heuristic from anchor grid extent
            img_w = int(anchor_pts[:, 0].max().item() + strides.max().item())
            img_h = int(anchor_pts[:, 1].max().item() + strides.max().item())

            fg_mask, assigned_gt, assigned_cls, assigned_box, assigned_iou = simota_assign(
                cls_logits[b_idx].detach(),
                bbox_decoded[b_idx].detach(),
                anchor_pts,
                gt_boxes,
                gt_labels,
                (img_h, img_w),
                topk_candidates=self.topk,
            )

            n_fg = fg_mask.sum().item()
            num_fg += n_fg

            # ── Classification (VFL on all anchors) ─────────────────────────
            target_score = torch.zeros(N, C, device=device)
            if n_fg > 0:
                # Soft label = IoU score at assigned class
                target_score[fg_mask, assigned_cls] = assigned_iou.float()

            total_cls = total_cls + varifocal_loss(
                cls_logits[b_idx], target_score,
                target_label=None,   # not used directly; target_score encodes it
            )

            if n_fg == 0:
                continue

            # ── BBox (CIoU on fg) ────────────────────────────────────────────
            pred_box_fg = bbox_decoded[b_idx][fg_mask]  # [n_fg, 4]
            total_bbox = total_bbox + ciou_loss(pred_box_fg, assigned_box).sum()

            # ── Keypoints (OKS, vis BCE on fg) ──────────────────────────────
            pred_kpts_fg = kpts_decoded[b_idx][fg_mask]    # [n_fg, K, 2]
            pred_vis_fg  = kpt_vis_raw[b_idx][fg_mask]     # [n_fg, K]

            gt_kpts_fg_xy  = gt_kpts[assigned_gt, :, :2]  # [n_fg, K, 2]
            gt_kpts_fg_vis = gt_kpts[assigned_gt, :, 2]   # [n_fg, K]
            gt_area_fg     = gt_area[assigned_gt]          # [n_fg]

            total_kpts = total_kpts + oks_loss_rtmo(
                pred_kpts_fg, gt_kpts_fg_xy, gt_kpts_fg_vis, gt_area_fg, self.sigmas
            ).sum()

            # Visibility BCE
            vis_target = (gt_kpts_fg_vis > 0).float()
            total_vis = total_vis + F.binary_cross_entropy_with_logits(
                pred_vis_fg, vis_target, reduction="sum"
            )

        # Normalise by total foreground (at least 1)
        normaliser = max(num_fg, 1)

        loss_dict = {
            "loss_cls":  total_cls  / normaliser * self.weight_dict["loss_cls"],
            "loss_bbox": total_bbox / normaliser * self.weight_dict["loss_bbox"],
            "loss_kpts": total_kpts / normaliser * self.weight_dict["loss_kpts"],
            "loss_vis":  total_vis  / normaliser * self.weight_dict["loss_vis"],
        }
        return loss_dict
