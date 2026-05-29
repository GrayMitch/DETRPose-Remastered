"""
DETRPose TensorRT inference script.
Matches the output style of tools/scripts/inference.py (PyTorch).

Usage:
    python tools/inference/trt_inf.py `
        --trt trt_engines/detrpose_hgnetv2_s_custom.engine `
        -i path/to/image_or_folder `
        -o predictions `
        --conf 0.35
"""
import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops.boxes import nms as torchvision_nms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from class_mapping_utils import find_class_mappings_json, print_detections


# ---------------------------------------------------------------------------
# Drawing helpers (mirrors tools/scripts/inference.py exactly)
# ---------------------------------------------------------------------------

def get_object_color(index):
    colors = [
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 0, 255),      # red
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (0, 165, 255),    # orange
        (128, 0, 255),    # purple
        (255, 128, 0),    # light blue
        (128, 255, 0),    # lime
        (180, 105, 255),  # pink-ish
        (42, 42, 165),    # brown-ish
    ]
    return colors[index % len(colors)]


def normalize_keypoints(kps, image_w, image_h):
    kps = np.asarray(kps)
    if kps.ndim == 1:
        kps = kps.reshape(-1, 2) if len(kps) % 2 == 0 else kps.reshape(-1, 3)[:, :2]
    elif kps.ndim == 2:
        kps = kps[:, :2]
    else:
        return []
    points = []
    for kp in kps:
        x, y = float(kp[0]), float(kp[1])
        if x <= 1.5 and y <= 1.5:
            x *= image_w
            y *= image_h
        x, y = int(round(x)), int(round(y))
        if 0 <= x < image_w and 0 <= y < image_h:
            points.append((x, y))
    return points


def draw_skeleton(img, kps, color, skeleton=None):
    h, w = img.shape[:2]
    points = normalize_keypoints(kps, w, h)
    for x, y in points:
        cv2.circle(img, (x, y), 3, color, -1)
        cv2.circle(img, (x, y), 4, (0, 0, 0), 1)
    if skeleton:
        for a, b in skeleton:
            if a < len(points) and b < len(points):
                cv2.line(img, points[a], points[b], color, 1)
    else:
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 1)
    return points


def draw_label(img, text, anchor_point, color):
    x, y = anchor_point
    h, w = img.shape[:2]
    x = max(0, min(int(x), w - 1))
    y = max(20, min(int(y), h - 1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    box_x1, box_y1 = x, y - text_h - baseline - 8
    box_x2, box_y2 = x + text_w + 10, y + baseline
    if box_x2 >= w:
        shift = box_x2 - w + 2
        box_x1 -= shift; box_x2 -= shift
    if box_y1 < 0:
        box_y1 = y; box_y2 = y + text_h + baseline + 8
        text_y = box_y1 + text_h + 4
    else:
        text_y = box_y2 - baseline - 4
    box_x1 = max(0, box_x1); box_y1 = max(0, box_y1)
    box_x2 = min(w - 1, box_x2); box_y2 = min(h - 1, box_y2)
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(img, text, (box_x1 + 5, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# TensorRT engine wrapper
# ---------------------------------------------------------------------------

class TRTInference:
    def __init__(self, engine_path, device="cuda:0", max_batch_size=1):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not installed. Run: pip install tensorrt")

        self.device = device
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_names  = [n for n in self.engine if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.engine if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.output_buffers = self._alloc_output_buffers(max_batch_size)
        self.stream = torch.cuda.Stream(device=device)

    def _alloc_output_buffers(self, max_batch_size):
        trt = self.trt
        buffers = {}
        for name in self.output_names:
            shape = list(self.engine.get_tensor_shape(name))
            if shape[0] == -1:
                shape[0] = max_batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            buffers[name] = data
        return buffers

    def __call__(self, blob):
        trt = self.trt

        # Set input shapes and addresses
        for n in self.input_names:
            tensor = blob[n]
            # Cast to the dtype the engine expects
            engine_dtype = trt.nptype(self.engine.get_tensor_dtype(n))
            tensor = tensor.to(dtype=torch.from_numpy(np.empty(0, dtype=engine_dtype)).dtype)
            tensor = tensor.contiguous().to(self.device)
            blob[n] = tensor
            self.context.set_input_shape(n, list(tensor.shape))
            self.context.set_tensor_address(n, tensor.data_ptr())

        # Set output addresses (reallocate if batch size changed)
        for n in self.output_names:
            out_shape = list(self.context.get_tensor_shape(n))
            if list(self.output_buffers[n].shape) != out_shape:
                dtype = trt.nptype(self.engine.get_tensor_dtype(n))
                self.output_buffers[n] = torch.from_numpy(
                    np.empty(out_shape, dtype=dtype)).to(self.device)
            self.context.set_tensor_address(n, self.output_buffers[n].data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        return {n: self.output_buffers[n] for n in self.output_names}


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

NMS_IOU_THRESHOLD = 0.65

class TRTInferenceRunner:
    def __init__(self, engine_path, conf_thresh=0.35, image_size=640, device="cuda:0"):
        self.conf_thresh = conf_thresh
        self.image_size = image_size
        self.device = device if torch.cuda.is_available() else "cpu"

        print(f"Loading TRT engine: {engine_path}")
        self.model = TRTInference(engine_path, device=self.device)
        print(f"Output tensors: {self.model.output_names}")
        self.has_boxes = 'boxes' in self.model.output_names

        self.class_mappings, self.skeleton_connections = find_class_mappings_json(engine_path)
        if not self.class_mappings:
            print("Warning: No class mappings found. Using numeric IDs.")
        if not self.skeleton_connections:
            print("Warning: No skeleton connections found. Using linear chain fallback.")

    def preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def infer_image(self, image_path, output_dir):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image: {image_path}")
            return

        h, w = img.shape[:2]
        im_data = self.preprocess(img)
        orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=self.device)

        outputs = self.model({"images": im_data, "orig_target_sizes": orig_size})

        scores    = outputs['scores'][0].cpu().numpy()      # (N,)
        labels    = outputs['labels'][0].cpu().numpy()      # (N,)
        keypoints = outputs['keypoints'][0].cpu().numpy()   # (N, K, 2)
        boxes     = outputs['boxes'][0].cpu().numpy() if self.has_boxes else None  # (N, 4)

        keep = scores > self.conf_thresh
        scores    = scores[keep]
        labels    = labels[keep]
        keypoints = keypoints[keep]
        if boxes is not None:
            boxes = boxes[keep]

        # Class-aware NMS (mirrors PostProcess non-deploy path)
        if boxes is not None and len(scores) > 0:
            t_boxes  = torch.from_numpy(boxes.astype(np.float32))
            t_scores = torch.from_numpy(scores.astype(np.float32))
            t_labels = torch.from_numpy(labels.astype(np.float32))
            max_coord = t_boxes.max()
            offsets = t_labels * (max_coord + 1)
            keep_nms = torchvision_nms(t_boxes + offsets[:, None], t_scores, NMS_IOU_THRESHOLD).numpy()
            scores    = scores[keep_nms]
            labels    = labels[keep_nms]
            keypoints = keypoints[keep_nms]
            boxes     = boxes[keep_nms]

        if len(scores) > 0:
            print(f"\nDetections in {image_path.name}:")
            print_detections(labels.astype(int), scores, self.class_mappings, max_display=10)

        vis = img.copy()
        for i, (score, label, kps) in enumerate(zip(scores, labels, keypoints)):
            color = get_object_color(i)
            class_name = self.class_mappings.get(int(label), f"class_{int(label)}")
            text = f"{class_name} {score:.3f}"

            skeleton = self.skeleton_connections.get(int(label), [])
            points = draw_skeleton(vis, kps, color, skeleton)

            if boxes is not None:
                x1, y1, x2, y2 = boxes[i][:4]
                if x2 <= 1.5 and y2 <= 1.5:
                    x1 *= w; x2 *= w; y1 *= h; y2 *= h
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                draw_label(vis, text, (x1, y1), color)
            elif points:
                xs = [p[0] for p in points]; ys = [p[1] for p in points]
                draw_label(vis, text, (max(0, min(xs)), max(20, min(ys) - 10)), color)

        output_path = output_dir / f"pred_{image_path.name}"
        cv2.imwrite(str(output_path), vis)
        print(f"-> {image_path.name}: {len(scores)} detections")

    def infer_path(self, input_path, output_dir):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            self.infer_image(input_path, output_dir)
            return

        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(input_path.rglob(ext))
        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"No images found in: {input_path}")
            return

        print(f"Found {len(image_paths)} images")
        for img_path in image_paths:
            self.infer_image(img_path, output_dir)


def resolve_and_extract_archives(input_path):
    """
    Resolves the input path before inference:
    - If input_path is a .7z or .zip archive, extract it to a same-named folder.
    - Then recursively scan the folder for any nested archives, extract and delete
      them until none remain.
    Returns the resolved folder Path.
    """
    import subprocess

    ARCHIVE_EXTS = {'.7z', '.zip'}

    def _extract_one(archive_path):
        archive_path = Path(archive_path)
        out_dir = archive_path.parent / archive_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ['7z', 'x', str(archive_path), f'-o{out_dir}', '-y'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"7z extraction failed for {archive_path.name}:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        archive_path.unlink()
        print(f"Extracted and removed archive: {archive_path.name} -> {out_dir}")
        return out_dir

    input_path = Path(input_path)
    # If the input itself is an archive, extract it to a folder first.
    if input_path.is_file() and input_path.suffix.lower() in ARCHIVE_EXTS:
        input_path = _extract_one(input_path)

    # Recursively extract any nested archives until none remain.
    if input_path.is_dir():
        while True:
            archives = sorted(
                p for p in input_path.rglob('*')
                if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS
            )
            if not archives:
                break
            for arch in archives:
                _extract_one(arch)

    return input_path


def main():
    parser = argparse.ArgumentParser(description="DETRPose TensorRT inference")
    parser.add_argument("--trt", type=str, required=True, help="Path to the TensorRT .engine file.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image or folder.")
    parser.add_argument("-o", "--output", type=str, default="predictions", help="Output directory (default: predictions).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (default: cuda:0).")
    parser.add_argument("--image-size", type=int, default=640, help="Input image size (default: 640).")
    args = parser.parse_args()

    infer = TRTInferenceRunner(
        engine_path=args.trt,
        conf_thresh=args.conf,
        image_size=args.image_size,
        device=args.device,
    )
    resolved_input = resolve_and_extract_archives(args.input)
    infer.infer_path(resolved_input, args.output)
    print(f"\nDone. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
