from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset_custom import dataset_train, dataset_val, dataset_test, evaluator 

from src.core import LazyCall as L
from src.nn.optimizer import ModelEMA 
from src.misc.get_param_dicts import get_optim_params

from torch import optim

# ── Output / training ────────────────────────────────────────────────────────
training_params.output_dir =  "output/detrpose_custom"
training_params.epochs = 100
training_params.use_ema = True
training_params.grad_accum_steps = 1

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

# ── Backbone (HGNetV2-S / B0) ─────────────────────────────────────────────
model.backbone.name = 'B0'
model.backbone.use_lab = True
model.encoder.in_channels = [256, 512, 1024]
model.encoder.depth_mult = 0.34
model.encoder.expansion = 0.5
model.transformer.num_decoder_layers = 3

# ── Custom dataset settings ───────────────────────────────────────────────
# 16 categories with IDs up to 23 → set num_classes = 24
# 11 keypoints per instance
NUM_CLASSES = 24
NUM_BODY_POINTS = 11

model.transformer.num_body_points = NUM_BODY_POINTS
model.transformer.num_classes = NUM_CLASSES

criterion.num_classes = NUM_CLASSES
criterion.num_body_points = NUM_BODY_POINTS
criterion.matcher.num_body_points = NUM_BODY_POINTS
criterion.matcher.cost_class = 2.0
criterion.matcher.cost_keypoints = 10.0
criterion.matcher.cost_oks = 4.0

postprocessor.num_body_points = NUM_BODY_POINTS

# ── Augmentation schedule (scaled for smaller dataset) ────────────────────
dataset_train.dataset.transforms.policy = {
    'name': 'stop_epoch',
    'ops': ['Mosaic', 'RandomCrop', 'RandomZoomOut'],
    'epoch': [5, 53, 96]
    }
dataset_train.collate_fn.base_size_repeat = 20
dataset_train.collate_fn.stop_epoch = 96
