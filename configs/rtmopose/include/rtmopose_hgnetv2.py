"""
RTMOPose base model/criterion/postprocessor definitions.
Imported by size-specific config files.
"""

from src.core import LazyCall as L
from src.models.rtmo import RTMOPose, RTMOCriterion, RTMOPostProcess
from src.nn import HGNetv2
import numpy as np

# Shared parameters (overridden by dataset-specific configs)
eval_spatial_size = (640, 640)
feat_strides = [8, 16, 32]
num_classes = 2
num_body_points = 17

training_params = {
    "clip_max_norm": 0.1,
    "save_checkpoint_interval": 1,
    "grad_accum_steps": 1,
    "print_freq": 100,
    "sync_bn": False,
    "use_ema": False,
    "dist_url": "env://",
}

# Default OKS sigmas for 17-keypoint COCO; overridden per dataset
_default_sigmas = [
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
    1.07, 1.07, .87, .87, .89, .89
]

postprocessor = L(RTMOPostProcess)(
    score_threshold=0.3,
    nms_threshold=0.65,
    max_detections=300,
    deploy_mode=False,
)

model = L(RTMOPose)(
    backbone=L(HGNetv2)(
        name="B0",
        use_lab=True,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=True,
        pretrained=True,
    ),
    neck_out_channels=256,
    neck_depth_mult=0.34,
    num_classes=num_classes,
    num_body_points=num_body_points,
    feat_strides=feat_strides,
    head_num_convs=2,
    act="silu",
    post_processor=postprocessor,
)

criterion = L(RTMOCriterion)(
    num_classes=num_classes,
    num_body_points=num_body_points,
    sigmas=_default_sigmas,
    weight_dict={
        "loss_cls":  1.0,
        "loss_bbox": 2.0,
        "loss_kpts": 5.0,
        "loss_vis":  1.0,
    },
    topk_candidates=13,
)
