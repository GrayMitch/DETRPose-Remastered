
<h2 align="center">
  DETRPose-Remastered: Real-time end-to-end transformer model for custom pose estimation
</h2>

<p align="center">
  <a href="https://github.com/SebastianJanampa/DETRPose/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>

  <a href="https://www.arxiv.org/abs/2506.13027">
        <img alt="arxiv" src="https://img.shields.io/badge/-paper-gray?style=for-the-badge&logo=arxiv&labelColor=red">
  </a>
</p>

<p align="center">
    📄 This repository is a fork and extension of the original DETRPose implementation:
    <br>
    <a href="https://www.arxiv.org/abs/2506.13027">DETRPose: Real-time end-to-end transformer model for multi-person pose estimation</a>
    <br>
    by Sebastian Janampa and Marios Pattichis (The University of New Mexico)
</p>

---

## Table of Contents
- [About This Fork](#about-this-fork)
- [Key Technical Features](#key-technical-features)
- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Data Preparation](#data-preparation)
  - [Usage](#usage)
- [Tools](#tools)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## About This Fork

This is an extended version of [DETRPose](https://github.com/SebastianJanampa/DETRPose) with added support for **custom datasets** and **custom keypoint configurations**. The original DETRPose is the first real-time end-to-end transformer model for multi-person pose estimation, achieving state-of-the-art results on COCO and CrowdPose datasets.

### New Features

- ✨ **Custom Dataset Support**: Train on your own pose estimation datasets with custom keypoint definitions
- 🎯 **Flexible Keypoint Configuration**: Define custom keypoint structures and skeleton connections via JSON
- 🎭 **Mask Annotation Support**: Support for segmentation masks alongside keypoint annotations
- 📦 **Automated Data Extraction**: Built-in support for .7z archives with automatic extraction
- 🔧 **Enhanced Inference Tools**: Improved inference scripts with custom category support
- 📊 **Custom Category Management**: Easy configuration of custom object categories

<p align="center">
    <a href="https://paperswithcode.com/sota/multi-person-pose-estimation-on-crowdpose?p=detrpose-real-time-end-to-end-transformer">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detrpose-real-time-end-to-end-transformer/multi-person-pose-estimation-on-crowdpose">
    </a>
</p>

## Key Technical Features

DETRPose introduces:
- **OKS-based Denoising**: A new denoising technique using Object Keypoint Similarity (OKS) metric for generating positive and negative queries
- **Advanced Classification Head**: Variations of the LQE head and varifocal loss adapted from D-FINE
- **Real-time Performance**: Achieves state-of-the-art accuracy while maintaining real-time inference speeds
- **End-to-end Architecture**: Fully transformer-based model without complex post-processing

> **Note**: For original DETRPose benchmarks and pre-trained weights on COCO2017 and CrowdPose, see the [original repository](https://github.com/SebastianJanampa/DETRPose). This fork focuses on custom dataset support with a simplified COCO format.

## Quick start

### Setup

```shell
conda create -n detrpose python=3.11.9
conda activate detrpose
pip install -r requirements.txt
```

### Data Preparation

This repository uses a simplified COCO format for custom pose estimation datasets:

```
data/
  └── coco/  # or your dataset name
      ├── Info.json              # Dataset metadata
      ├── kpts_definition.json   # Keypoint definitions and skeleton
      ├── train/
      │   ├── coco_instances.json
      │   ├── images/
      │   └── masks/            # Optional: segmentation masks
      │       ├── 00000000.json
      │       ├── 00000000.png
      │       └── ...
      └── val/
          ├── coco_instances.json
          ├── images/
          └── masks/            # Optional: segmentation masks
```

**Custom Dataset Configuration**:

1. **kpts_definition.json**: Define your keypoint structure
```json
{
  "categories": [
    {
      "super_category": "default",
      "category": "your_category_name",
      "keypoint_count": 11,
      "keypoints_names": ["kp0", "kp1", "kp2", ...],
      "keypoint_connection_rules": {
        "skeleton": [["kp0", "kp1"], ["kp1", "kp2"], ...]
      }
    }
  ]
}
```

2. **Info.json**: Basic dataset metadata
```json
{
  "description": "Your dataset description",
  "version": "1.0"
}
```

3. **coco_instances.json**: COCO-format annotations with custom keypoints

4. **Configure the model**: Use or modify `configs/detrpose/detrpose_hgnetv2_*_custom.py` files to match your dataset settings.

**Key Features of Custom Dataset Support**:
- 📊 **Automatic Parameter Detection**: NUM_CLASSES and NUM_BODY_POINTS are automatically derived from your annotations
- 🗜️ **7z Archive Support**: Automatically extracts `.7z` archives for train/val folders if directories don't exist
- 🎯 **Flexible Keypoint Structures**: Support for any number of keypoints and custom skeleton definitions
- 🎨 **Mask Annotations**: Optional support for segmentation masks alongside keypoint annotations
- 🔄 **Multiple Categories**: Support for multiple object categories with different keypoint structures
- ⚙️ **Easy Configuration**: Simple JSON-based configuration files for dataset metadata

### Usage

#### Training

**Linux/Mac:**
```bash
# Choose model size: n, s, m, l, or x
export MODEL=s

# Single GPU training
python train.py \
  --config_file configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py \
  --device cuda \
  --amp

# Multi-GPU training (4 GPUs example)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7777 \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py \
  --device cuda \
  --amp
```

**Windows (PowerShell):**
```powershell
# Choose model size: n, s, m, l, or x
$env:MODEL="s"

# Single GPU training
python train.py `
  --config_file configs/detrpose/detrpose_hgnetv2_$($env:MODEL)_custom.py `
  --device cuda `
  --amp

# Multi-GPU training (4 GPUs example)
torchrun `
  --master_port=7777 `
  --nproc_per_node=4 `
  train.py `
  --config_file configs/detrpose/detrpose_hgnetv2_$($env:MODEL)_custom.py `
  --device cuda `
  --amp
```

#### Evaluation

**Linux/Mac:**
```bash
# Evaluate on validation set
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7777 \
  --nproc_per_node=4 \
  train.py \
  --config_file configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py \
  --device cuda \
  --amp \
  --resume output/detrpose_hgnetv2_${MODEL}_custom/checkpoint_best_regular.pth \
  --eval
```

**Windows (PowerShell):**
```powershell
# Evaluate on validation set
torchrun `
  --master_port=7777 `
  --nproc_per_node=4 `
  train.py `
  --config_file configs/detrpose/detrpose_hgnetv2_$($env:MODEL)_custom.py `
  --device cuda `
  --amp `
  --resume output/detrpose_hgnetv2_$($env:MODEL)_custom/checkpoint_best_regular.pth `
  --eval
```

#### Inference

```bash
# Run inference on images or videos
python tools/scripts/inference.py \
  --checkpoint output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --input path/to/image_or_folder \
  --output predictions/ \
  --conf_thresh 0.35 \
  --device cuda
```

## Tools

### Inference Scripts

This repository includes multiple inference options:

<details>
<summary> Custom Inference Script </summary>

The `tools/scripts/inference.py` provides an easy-to-use interface for custom datasets:

```shell
python tools/scripts/inference.py \
    --checkpoint output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
    --input path/to/image_or_folder \
    --output predictions/ \
    --conf_thresh 0.35 \
    --device cuda
```

**Features**:
- Supports both images and folders
- Automatic EMA weight loading
- Configurable confidence threshold
- Custom category name mapping
- Color-coded visualization

</details>

<details>
<summary> Deployment </summary>

1. Setup
```shell
pip install -r tools/inference/requirements.txt
export MODEL=s  # n s m l x
```

2. Export ONNX
```shell
python tools/deployment/export_onnx.py \
  --check \
  -c configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py \
  -r output/detrpose_hgnetv2_${MODEL}_custom/checkpoint_best_regular.pth
```

3. Export [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

For a specific file:
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

For all files in a folder:
```shell
python tools/deployment/export_tensorrt.py
```

</details>

<details>
<summary> ONNX/TensorRT Inference </summary>

Inference on images and videos is supported through ONNX and TensorRT backends.

```shell
# ONNX inference on a single file
python tools/inference/onnx_inf.py \
  --onnx detrpose_hgnetv2_${MODEL}_custom.onnx \
  --input examples/example1.jpg \
  --annotator COCO

# TensorRT inference on a single file
python tools/inference/trt_inf.py \
  --trt detrpose_hgnetv2_${MODEL}_custom.engine \
  --input examples/example1.jpg \
  --annotator COCO

# PyTorch inference on a single file
python tools/inference/torch_inf.py \
  -c configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py \
  -r output/detrpose_hgnetv2_${MODEL}_custom/checkpoint_best_regular.pth \
  --input examples/example1.jpg \
  --device cuda:0

# Batch inference on a folder
python tools/inference/onnx_inf.py \
  --onnx detrpose_hgnetv2_${MODEL}_custom.onnx \
  --input examples/ \
  --annotator COCO
```

</details>

<details>
<summary> Benchmark </summary>

1. Setup
```shell
pip install -r tools/benchmark/requirements.txt
export MODEL=s  # n s m l x
```

2. Model FLOPs, MACs, and Params
```shell
python tools/benchmark/get_info.py \
  --config configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py
```

3. TensorRT Latency
```shell
python tools/benchmark/trt_benchmark.py \
  --infer_dir ./data/coco/val/images \
  --engine_dir trt_engines
```

4. PyTorch Latency
```shell
python tools/benchmark/torch_benchmark.py \
  -c ./configs/detrpose/detrpose_hgnetv2_${MODEL}_custom.py \
  --resume output/detrpose_hgnetv2_${MODEL}_custom/checkpoint_best_regular.pth \
  --infer_dir ./data/coco/val/images
```
</details>


## Citation

### Original DETRPose Paper
If you use DETRPose or its methods in your work, please cite the original paper:
<details open>
<summary> BibTeX </summary>

```latex
@misc{janampa2025detrpose,
      title={DETRPose: Real-time end-to-end transformer model for multi-person pose estimation}, 
      author={Sebastian Janampa and Marios Pattichis},
      year={2025},
      eprint={2506.13027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.13027}, 
}
```
</details>

### This Repository
If you use the custom dataset features or enhancements from this fork, please also link to this repository:
```
https://github.com/GrayMitch/DETRPose-Remastered
```

## Acknowledgement

This repository is a fork and extension of [DETRPose](https://github.com/SebastianJanampa/DETRPose) by Sebastian Janampa and Marios Pattichis from The University of New Mexico.

The original DETRPose work was supported in part by [Lambda.ai](https://lambda.ai).

The original DETRPose implementation builds upon [DEIM](https://github.com/Intellindust-AI-Lab/DEIM/tree/main), [D-FINE](https://github.com/Peterande/D-FINE), [Detectron2](https://github.com/facebookresearch/detectron2/tree/main), and [GroupPose](https://github.com/Michel-liu/GroupPose/tree/main).

### Additional Features
The custom dataset support and enhancements in this fork extend the original implementation to support flexible pose estimation on custom datasets with arbitrary keypoint configurations.

✨ Feel free to contribute and reach out if you have any questions! ✨

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Note**: This is a fork of the original DETRPose repository. Please ensure you comply with the licenses of all dependencies and the original work when using this code.
