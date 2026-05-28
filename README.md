
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
- [Model Zoo](#model-zoo)
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

## Model Zoo

The original DETRPose weights are available from the [original repository](https://github.com/SebastianJanampa/DETRPose/releases). These weights achieve state-of-the-art performance on standard benchmarks.

### COCO val2017
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 57.2 | 81.7 | 61.4 | 64.4 | 87.9 | 4.1 M | 2.80 ms | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n.py) | [57.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n.pth) | 
**DETRPose-S** | 67.0 | 87.6 | 72.8 | 73.5 | 92.4 | 11.5 M | 4.99 ms | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s.py) | [67.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth) | 
**DETRPose-M** | 69.4 | 89.2 | 75.4 | 75.5 | 93.7 | 20.8 M | 7.01 ms | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m.py) | [69.4](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth) | 
**DETRPose-L** | 72.5 | 90.6 | 79.0 | 78.7 | 95.0 | 32.8 M | 9.50 ms | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l.py) | [72.5](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth) | 
**DETRPose-X** | 73.3 | 90.5 | 79.4 | 79.4 | 94.9 | 73.3 M | 13.31 ms | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x.py) | [73.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x.pth) | 

### COCO test-dev2017
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 56.7 | 83.1 | 61.1 | 64.4 | 89.3 | 4.1 M | 2.80 ms | 9.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n.py) | [56.7](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n.pth) | 
**DETRPose-S** | 66.0 | 88.3 | 72.0 | 73.2 | 93.3 | 11.5 M | 4.99 ms | 33.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s.py) | [66.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s.pth) | 
**DETRPose-M** | 68.4 | 90.1 | 74.8 | 75.1 | 94.4 | 20.8 M | 7.01 ms | 67.3 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m.py) | [88.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m.pth) | 
**DETRPose-L** | 71.2 | 91.2 | 78.1 | 78.1 | 95.7 | 32.8 M | 9.50 ms | 107.1 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l.py) | [71.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l.pth) | 
**DETRPose-X** | 72.2 | 91.4 | 79.3 | 78.8 | 95.7 | 73.3 M | 13.31 ms | 239.5 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x.py) | [72.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x.pth) | 

### CrowdPose test
| Model  | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>E</sup> | AP<sup>M</sup> | AP<sup>H</sup> | #Params | Latency | GFLOPs | config | checkpoint |
| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
**DETRPose-N** | 56.0 | 80.7 | 59.6 | 65.0 | 56.6 | 46.6 | 4.1 M | 2.72 ms | 8.8 | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_n_crowdpose.py) | [57.2](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_n_crowdpose.pth) | 
**DETRPose-S** | 67.4 | 88.6 | 72.9 | 74.7 | 68.1 | 59.3 | 11.5 M | 4.80 ms | 31.3  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_s_crowdpose.py) | [67.0](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_s_crowdpose.pth) | 
**DETRPose-M** | 72.0 | 91.0 | 77.8 | 78.6 | 72.6 | 64.5 | 20.7 M | 6.86 ms | 64.9  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_m_crowdpose.py) | [69.4](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_m_crowdpose.pth) | 
**DETRPose-L** | 73.3 | 91.6 | 79.4 | 79.5 | 74.0 | 66.1 | 32.7 M | 9.03 ms | 103.5  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_l_crowdpose.py) | [72.5](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_l_crowdpose.pth) | 
**DETRPose-X** | 75.1 | 92.1 | 81.3 | 81.3 | 75.7 | 68.1 | 73.3 M | 13.01 ms | 232.3  | [py](https://github.com/SebastianJanampa/DETRPose/blob/main/configs/detrpose/detrpose_hgnetv2_x_crowdpose.py) | [73.3](https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_x_crowdpose.pth) | 

**Notes:**
- **Latency** is evaluated on a single  Tesla V100 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==8.6.3$.

## Quick start

### Setup

```shell
conda create -n detrpose python=3.11.9
conda activate detrpose
pip install -r requirements.txt
```

### Data Preparation

#### Option 1: Standard Datasets (COCO/CrowdPose)

Create a folder named `data` to store the datasets
```
data/
  ├── COCO2017
  │   ├── train2017
  │   ├── val2017
  │   ├── test2017
  │   └── annotations
  └── crowdpose
      ├── images
      └── annotations
```

<details>
  <summary> COCO2017 dataset </summary>
  Download COCO2017 from their [website](https://cocodataset.org/#download)
</details>
<details>
  <summary> CrowdPose dataset </summary>
  Download Crowdpose from their [github](https://github.com/jeffffffli/CrowdPose), or use the following command
  
```shell
pip install gdown # to download files from google drive
mkdir crowdpose
cd crowdpose
gdown 1VprytECcLtU4tKP32SYi_7oDRbw7yUTL # images
gdown 1b3APtKpc43dx_5FxizbS-EWGvd-zl7Lb # crowdpose_train.json
gdown 18-IwNa6TOGQPE0RqGNjNY1cJOfNC7MXj # crowdpose_val.json
gdown 13xScmTWqO6Y6m_CjiQ-23ptgX9sC-J9I # crowdpose_trainval.json
gdown 1FUzRj-dPbL1OyBwcIX2BgFPEaY5Yrz7S # crowdpose_test.json
unzip images.zip
```
</details>

#### Option 2: Custom Dataset

For training on your own custom pose estimation dataset:

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

**Note**: The examples below use Unix-style commands. For Windows users:
- Replace `export model=s` with `$env:model="s"` (PowerShell) or `set model=s` (CMD)
- Use backslashes `\` instead of forward slashes `/` for paths if needed
- You can use `torchrun` directly without `CUDA_VISIBLE_DEVICES` prefix

<details open>
  <summary> Custom Dataset </summary>
  
1. Set Model
```shell
export model=s # n s m l x
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_custom.py --device cuda --amp --pretrain dfine_${model}_obj365 
```

3. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_custom.py --device cuda --amp --resume <PTH_FILE_PATH> --eval
```

4. Inference
```shell
python tools/scripts/inference.py --checkpoint <PTH_FILE_PATH> --input <IMAGE_OR_FOLDER> --output <OUTPUT_FOLDER>
```
</details>

<details>
  <summary> COCO2017 dataset </summary>
  
1. Set Model
```shell
export model=l # n s m l x
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --pretrain dfine_${model}_obj365 
```
if you choose `model=n`, do
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_n.py --device cuda --amp --pretrain dfine_n_obj365 
```

3. Testing (COCO2017 val)
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --resume <PTH_FILE_PATH> --eval
```

4. Testing (COCO2017 test-dev)
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --resume <PTH_FILE_PATH> --test
```
After running the command. You'll get a file named `results.json`. Compress it and submit it to the [COCO competition website](https://codalab.lisn.upsaclay.fr/competitions/7403#learn_the_details)

5. Replicate results (optional)
```shell
# First, download the official weights from the original DETRPose repository
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_${model}.pth

# Second, run evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --resume detrpose_hgnetv2_${model}.pth --eval
```
</details>

<details>
  <summary> CrowdPose dataset </summary>
  
1. Set Model
```shell
export model=l # n s m l x
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --device cuda --amp --pretrain dfine_${model}_obj365 
```
if you choose `model=n`, do
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_n_crowdpose.py --device cuda --amp --pretrain dfine_n_obj365 
```

3. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --device cuda --amp --resume <PTH_FILE_PATH> --eval
```

4. Replicate results (optional)
```shell
# First, download the official weights from the original DETRPose repository
wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_${model}_crowdpose.pth

# Second, run evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4  train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --device cuda --amp --resume detrpose_hgnetv2_${model}_crowdpose.pth --eval
```
</details>

### Lambda instances
All latency experiments using Lambda.ai instances. We have provided two README files 

  1. to run a [TensorRT container ](https://github.com/SebastianJanampa/DETRPose/blob/main/assets/TENSORRT_CONTAINER_LAMBDA.AI.md)in a Lambda.ai instance 
  2. to install a [TensorRT `.deb`](https://github.com/SebastianJanampa/DETRPose/blob/main/assets/TENSORRT_DEB_LAMBDA.AI.md) in a Lambda.ai instance

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

<!-- <summary>4. Export onnx </summary> -->
1. Setup
```shell
pip install -r tools/inference/requirements.txt
export model=l  # n s m l x
```

2. Export onnx
For COCO model
```shell
python tools/deployment/export_onnx.py --check -c configs/detrpose/detrpose_hgnetv2_${model}.py -r detrpose_hgnetv2_${model}.pth
```

For CrowdPose model
```shell
python tools/deployment/export_onnx.py --check -c configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py -r detrpose_hgnetv2_${model}_crowdpose.pth
```

3. Export [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
For a specific file
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

or, for all files inside a folder
```shell
python tools/deployment/export_tensorrt.py
```

</details>

<details>
<summary> Inference (Visualization) </summary>


1. Setup
```shell
export model=l  # n s m l x
```


<!-- <summary>5. Inference </summary> -->
2. Inference (onnxruntime / tensorrt / torch)

Inference on images and videos is supported.

For a single file
```shell
# For COCO model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}.onnx --input examples/example1.jpg --annotator COCO
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}.engine --input examples/example1.jpg --annotator COCO
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}.py -r <checkpoint.pth> --input examples/example1.jpg --device cuda:0 

# For CrowdPose model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}_crowdpose.onnx --input examples/example1.jpg --annotator CrowdPose
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}_crowdpose.engine --input examples/example1.jpg --annotator CrowdPose
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py -r <checkpoint.pth> --input examples/example1.jpg --device cuda:0 
```

For a folder
```shell
# For COCO model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}.onnx --input examples --annotator COCO
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}.engine --input examples --annotator COCO
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}.py -r <checkpoint.pth> --input examples --device cuda:0 

# For CrowdPose model
python tools/inference/onnx_inf.py --onnx detrpose_hgnetv2_${model}_crowdpose.onnx --input examples --annotator CrowdPose
python tools/inference/trt_inf.py --trt detrpose_hgnetv2_${model}_crowdpose.engine --input examples --annotator CrowdPose
python tools/inference/torch_inf.py -c configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py -r <checkpoint.pth> --input examples --device cuda:0

```
</details>

<details>
<summary> Benchmark </summary>

1. Setup
```shell
pip install -r tools/benchmark/requirements.txt
export model=l  # n s m l
```

<!-- <summary>6. Benchmark </summary> -->
2. Model FLOPs, MACs, and Params
```shell
# For COCO model
python tools/benchmark/get_info.py --config configs/detrpose/detrpose_hgnetv2_${model}.py

# For COCO model
python tools/benchmark/get_info.py --config configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py
```

3. TensorRT Latency
```shell
python tools/benchmark/trt_benchmark.py --infer_dir ./data/COCO2017/val2017 --engine_dir trt_engines
```

4. Pytorch Latency
```shell
# For COCO model
python tools/benchmark/torch_benchmark.py -c ./configs/detrpose/detrpose_hgnetv2_${model}.py --resume detrpose_hgnetv2_${model}.pth --infer_dir ./data/COCO/val2017

# For CrowdPose model
python tools/benchmark/torch_benchmark.py -c ./configs/detrpose/detrpose_hgnetv2_${model}_crowdpose.py --resume detrpose_hgnetv2_${model}_crowdpose.pth --infer_dir ./data/COCO/val2017
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
