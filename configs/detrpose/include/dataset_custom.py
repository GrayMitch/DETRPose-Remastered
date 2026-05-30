import os
import json

from src.core import LazyCall as L
from src.data import CocoDetection
from src.data.dataloader import (
	BatchImageCollateFunction,
	DataLoader
	)
from src.data.coco_eval import CocoEvaluator
from src.data.container import Compose
import src.data.transforms as T

from .detrpose_hgnetv2 import eval_spatial_size

scales = [(640, 640)]
max_size = 640

__all__ = ["dataset_train", "dataset_val", "dataset_test", "evaluator", "NUM_CLASSES", "NUM_BODY_POINTS", "CLASS_MAPPINGS", "CLASS_SKELETONS"]

# ── Data root ─────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("DETRPOSE_DATA_ROOT", "/content/coco_data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")


def _extract_7z(archive_path: str, extract_to: str) -> None:
	"""Extract a .7z archive into extract_to using the native 7z CLI."""
	import subprocess
	result = subprocess.run(
		['7z', 'x', archive_path, f'-o{extract_to}', '-y'],
		capture_output=True, text=True
	)
	if result.returncode != 0:
		raise RuntimeError(
			f"Failed to extract '{archive_path}'.\n"
			"7-Zip must be installed and '7z' must be on PATH.\n"
			f"stderr: {result.stderr}"
		)


def _ensure_dir(dir_path: str) -> None:
	"""If dir_path does not exist as a directory, extract dir_path.7z."""
	if os.path.isdir(dir_path):
		return
	archive = dir_path + ".7z"
	if os.path.isfile(archive):
		print(f"[dataset_custom] Extracting '{archive}' ...")
		_extract_7z(archive, os.path.dirname(dir_path))
		if not os.path.isdir(dir_path):
			raise RuntimeError(
				f"Extraction of '{archive}' succeeded but '{dir_path}' still not found. "
				"Check that the archive contains a top-level folder named "
				f"'{os.path.basename(dir_path)}'."
			)
	else:
		raise FileNotFoundError(
			f"Neither directory '{dir_path}' nor archive '{archive}' found."
		)


_ensure_dir(TRAIN_DIR)
_ensure_dir(VAL_DIR)

TRAIN_ANN = os.path.join(TRAIN_DIR, "coco_instances.json")
VAL_ANN   = os.path.join(VAL_DIR,   "coco_instances.json")
TRAIN_IMG = os.path.join(TRAIN_DIR, "images")
VAL_IMG   = os.path.join(VAL_DIR,   "images")

# ── Dynamically derive dataset parameters ────────────────────────────────────
with open(TRAIN_ANN) as _f:
	_ann = json.load(_f)

_cats = _ann["categories"]
# num_classes must cover the highest category ID (IDs may be non-contiguous)
NUM_CLASSES     = max(c["id"] for c in _cats) + 1
NUM_BODY_POINTS = max(len(c["keypoints"]) for c in _cats)

# Create class mapping dictionary: {class_id: class_name}
CLASS_MAPPINGS = {c["id"]: c["name"] for c in _cats}

# Skeleton connections per class: {class_id: [[kp_a, kp_b], ...]}
CLASS_SKELETONS = {c["id"]: c.get("skeleton", []) for c in _cats}

# ─────────────────────────────────────────────────────────────────────────────

dataset_train = L(DataLoader)(
	dataset=L(CocoDetection)(
		img_folder=TRAIN_IMG,
		ann_file=TRAIN_ANN,
		transforms=L(Compose)(
			transforms1=L(T.RandomResize)(sizes=scales, max_size=max_size),
			transforms2=L(T.ToTensor)(),
			transforms3=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1])
			),
		),
	total_batch_size=8,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=8,
	shuffle=True,
	drop_last=True,
	pin_memory=True,
	persistent_workers=True,
	prefetch_factor=4,
	)

dataset_val = L(DataLoader)(
	dataset=L(CocoDetection)(
		img_folder=VAL_IMG,
		ann_file=VAL_ANN,
		transforms=L(Compose)(
			transforms1=L(T.RandomResize)(sizes=[eval_spatial_size], max_size=max_size),
			transforms2=L(T.ToTensor)(),
			transforms3=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1])
			),
		),
	total_batch_size=8,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=8,
	shuffle=False,
	drop_last=False,
	pin_memory=True,
	persistent_workers=True,
	prefetch_factor=4,
	)

dataset_test = L(DataLoader)(
	dataset=L(CocoDetection)(
		img_folder=VAL_IMG,
		ann_file=VAL_ANN,
		transforms=L(Compose)(
			transforms1=L(T.RandomResize)(sizes=[eval_spatial_size], max_size=max_size),
			transforms2=L(T.ToTensor)(),
			transforms3=L(T.Normalize)(mean=[0, 0, 0], std=[1, 1, 1])
			),
		),
	total_batch_size=8,
	collate_fn=L(BatchImageCollateFunction)(
		base_size=eval_spatial_size[0],
		),
	num_workers=8,
	shuffle=False,
	drop_last=False,
	pin_memory=True,
	persistent_workers=True,
	prefetch_factor=4,
	)

evaluator = L(CocoEvaluator)(
	ann_file=VAL_ANN,
	iou_types=['bbox', 'keypoints'],
	useCats=True
	)
