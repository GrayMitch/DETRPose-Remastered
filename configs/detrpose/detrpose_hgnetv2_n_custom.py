from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset_custom import (
    dataset_train, dataset_val, dataset_test, evaluator,
    NUM_CLASSES, NUM_BODY_POINTS, CLASS_MAPPINGS, CLASS_SKELETONS
)

from src.core import LazyCall as L
from src.nn.optimizer import ModelEMA
from src.misc.get_param_dicts import get_optim_params

from torch import optim

# ── Output / training ────────────────────────────────────────────────────────
training_params["output_dir"] = "output/detrpose_hgnetv2_n_custom"
training_params["epochs"] = 100
training_params["use_ema"] = True
training_params["grad_accum_steps"] = 1

ema = L(ModelEMA)(
    decay=0.9999,
    warmups=2000
    )

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(
        cfg=[
                {
                'params': '^(?=.*backbone).*$',
                'lr': 0.0001
                },
            ],
        ),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001
    )

lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(
    milestones=[1000],
    gamma=0.1
    )

# ── Backbone (HGNetV2-N / B0 nano) ────────────────────────────────────────
model.backbone.name = 'B0'
model.backbone.use_lab = True
model.backbone.return_idx = [2, 3]
model.encoder.in_channels = [512, 1024]
model.encoder.feat_strides = [16, 32]
model.encoder.n_levels = 2
model.encoder.use_encoder_idx = [1]
model.encoder.depth_mult = 0.5
model.encoder.expansion = 0.34
model.encoder.hidden_dim = 128
model.encoder.dim_feedforward = 512
model.transformer.num_decoder_layers = 3
model.transformer.num_feature_levels = 2
model.transformer.dim_feedforward = 512
model.transformer.feat_strides = [16, 32]
model.transformer.hidden_dim = 128
model.transformer.dec_n_points = 6

# ── Dataset-derived parameters (read from data/coco/train/coco_instances.json)
model.transformer.num_body_points = NUM_BODY_POINTS
model.transformer.num_classes = NUM_CLASSES

criterion.num_classes = NUM_CLASSES
criterion.num_body_points = NUM_BODY_POINTS
criterion.matcher.num_body_points = NUM_BODY_POINTS
criterion.matcher.cost_class = 2.0
criterion.matcher.cost_keypoints = 10.0
criterion.matcher.cost_oks = 4.0

postprocessor.num_body_points = NUM_BODY_POINTS


