"""
SimOTA: GPU-native label assignment for single-stage detectors.
Inspired by RTMO / RTMDet (Apache 2.0, OpenMMLab).
Custom standalone implementation – no MMDet dependency.

Key properties:
  * Runs entirely on CUDA (no .cpu() / scipy calls in the hot path)
  * Dynamic-k selection per GT (based on top-k IoU sum)
  * Conflict resolution: lowest cost wins
"""

import torch
import torch.nn.functional as F
from src.misc.box_ops import box_iou


def _sigmoid_focal_loss_single(pred_logits, label, alpha=0.25, gamma=2.0):
    """Binary focal loss: [N, C] logits, [N, C] labels → scalar."""
    prob = pred_logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(pred_logits, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    alpha_t = alpha * label + (1 - alpha) * (1 - label)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.sum(-1)   # [N] – summed over classes


def _iou_cost(boxes_pred, boxes_gt):
    """Compute 1 - IoU cost matrix. boxes in xyxy format. Returns [N_pred, N_gt]."""
    iou, _ = box_iou(boxes_pred, boxes_gt)
    return 1.0 - iou


@torch.no_grad()
def simota_assign(
    cls_logits,    # [N, num_classes]  – all anchor predictions in image
    bbox_xyxy,     # [N, 4]           – decoded predictions (absolute, xyxy)
    anchor_pts,    # [N, 2]           – anchor point xy
    gt_boxes,      # [G, 4]           – ground-truth boxes (xyxy absolute)
    gt_labels,     # [G]              – ground-truth class indices
    img_hw,        # (H, W)           – image size (for filtering outside points)
    topk_candidates=13,
    alpha=0.25,
    gamma=2.0,
    lambda_cls=1.0,
    lambda_iou=3.0,
):
    """
    Assign ground-truth targets to anchor points.

    Returns:
        fg_mask:      [N]        – bool mask of foreground anchors
        assigned_gt:  [N_fg]    – GT index assigned to each foreground anchor
        assigned_cls: [N_fg]    – GT class index per foreground anchor
        assigned_box: [N_fg, 4] – GT box per foreground anchor (xyxy)
    """
    N = cls_logits.shape[0]
    G = gt_boxes.shape[0]
    device = cls_logits.device

    if G == 0:
        fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
        dummy = torch.zeros(0, dtype=torch.long, device=device)
        return fg_mask, dummy, dummy, gt_boxes[:0], None

    # ── 1. Candidate anchors: inside GT boxes + within a centre region ──────
    # A point is a candidate if it falls inside the GT box
    # Shape: [N, G]
    ax = anchor_pts[:, 0:1]  # [N, 1]
    ay = anchor_pts[:, 1:2]  # [N, 1]

    gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    in_box = (
        (ax >= gt_x1[None]) & (ax <= gt_x2[None]) &
        (ay >= gt_y1[None]) & (ay <= gt_y2[None])
    )  # [N, G]

    # Additionally require anchor inside image
    H, W = img_hw
    in_image = (
        (anchor_pts[:, 0] >= 0) & (anchor_pts[:, 0] < W) &
        (anchor_pts[:, 1] >= 0) & (anchor_pts[:, 1] < H)
    )  # [N]
    in_box = in_box & in_image[:, None]

    # ── 2. Cost matrix for candidates ───────────────────────────────────────
    # Use -IoU + focal_cls_cost
    # IoU: [N, G]
    iou_mat, _ = box_iou(bbox_xyxy, gt_boxes)  # [N, G]

    # Focal cls cost: for each anchor–GT pair use the GT class one-hot
    one_hot = torch.zeros(N, G, cls_logits.shape[1], device=device)
    one_hot.scatter_(2, gt_labels[None, :, None].expand(N, G, 1), 1.0)  # [N, G, C]
    cls_expanded = cls_logits[:, None, :].expand(N, G, -1)               # [N, G, C]
    cls_cost = _sigmoid_focal_loss_single(cls_expanded.reshape(-1, cls_logits.shape[1]),
                                          one_hot.reshape(-1, cls_logits.shape[1])
                                          ).reshape(N, G)                # [N, G]

    cost = lambda_cls * cls_cost - lambda_iou * iou_mat  # [N, G]

    # Mask out non-candidates with large cost
    cost[~in_box] = 1e8

    # ── 3. Dynamic-k selection ───────────────────────────────────────────────
    # For each GT, select top-k anchors by IoU, sum those IoUs → dynamic_k
    n_candidates = in_box.sum(0).clamp(min=1)  # [G]
    k = topk_candidates

    # top-k IoU per GT (ignore non-candidates by masking)
    iou_for_k = iou_mat.clone()
    iou_for_k[~in_box] = 0.0
    topk_iou, _ = iou_for_k.topk(min(k, N), dim=0, largest=True)  # [k, G]
    dynamic_k = topk_iou.sum(0).clamp(min=1).int()                 # [G]

    # ── 4. Select anchors per GT using dynamic_k ────────────────────────────
    # sort cost per GT, take the lowest dynamic_k[g] indices
    sorted_cost, sorted_idx = cost.sort(0)  # [N, G]

    assigned = torch.full((N,), -1, dtype=torch.long, device=device)

    for g in range(G):
        dk = dynamic_k[g].item()
        sel = sorted_idx[:dk, g]   # top-dk anchor indices for GT g
        # Conflict resolve below; tentatively mark
        for idx in sel:
            existing = assigned[idx].item()
            if existing == -1:
                assigned[idx] = g
            else:
                # Keep the assignment with lower cost
                if cost[idx, g].item() < cost[idx, existing].item():
                    assigned[idx] = g

    fg_mask = (assigned >= 0)
    assigned_gt = assigned[fg_mask]
    assigned_cls = gt_labels[assigned_gt]
    assigned_box = gt_boxes[assigned_gt]
    assigned_iou = iou_mat[fg_mask, assigned_gt]

    return fg_mask, assigned_gt, assigned_cls, assigned_box, assigned_iou
