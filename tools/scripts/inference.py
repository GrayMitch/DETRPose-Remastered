import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ==================== PATH SETUP ====================
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

# Try both possible instantiate locations used by DETRPose-style repos
try:
    from src.core import instantiate
except ImportError:
    from src.core.workspace import instantiate


CLASS_NAMES = {
    2: "black_bx_side",
    3: "black_bk_side",
    4: "black_bk_upright",
    7: "black_bx_upright",
    10: "black_ar_side",
    11: "black__g_side",
    12: "black__h_upright",
    13: "white_ax_side",
    14: "black_az_side",
    16: "green_ar_upright",
    17: "green_ar_side",
    19: "white_ax_upright",
    20: "black_ar_upright",
    21: "black__h_side",
    22: "black__g_upright",
    23: "black_az_upright",
}


class DETRPoseInference:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        conf_thresh=0.35,
        config_module="configs.detrpose.detrpose_hgnetv2_s_custom",
        use_ema=True,
        image_size=640,
    ):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.conf_thresh = conf_thresh
        self.image_size = image_size

        print(f"Using device: {self.device}")
        print(f"Loading config: {config_module}")

        # Import as a package module so relative imports inside the config work.
        cfg = importlib.import_module(config_module)

        # Build the exact model/postprocessor from the same config used for training.
        self.model = instantiate(cfg.model)
        self.postprocessor = instantiate(cfg.postprocessor)

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = self._extract_state_dict(ckpt, use_ema=use_ema)

        # Remove possible DataParallel/DDP prefix.
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

        print("Loading model weights...")
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded successfully!")

    def _extract_state_dict(self, ckpt, use_ema=True):
        """
        Handles common checkpoint formats:
        - ckpt["ema"]["module"]
        - ckpt["ema"]["state_dict"]
        - ckpt["ema"]
        - ckpt["model"]
        - raw state_dict
        """
        if use_ema and isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"] is not None:
            ema = ckpt["ema"]

            if isinstance(ema, dict):
                if "module" in ema:
                    print("Using EMA weights: ckpt['ema']['module']")
                    return ema["module"]

                if "state_dict" in ema:
                    print("Using EMA weights: ckpt['ema']['state_dict']")
                    return ema["state_dict"]

                print("Using EMA weights: ckpt['ema']")
                return ema

        if isinstance(ckpt, dict) and "model" in ckpt:
            print("Using model weights: ckpt['model']")
            return ckpt["model"]

        print("Using raw checkpoint as state_dict")
        return ckpt

    def preprocess(self, image):
        """
        Manual preprocessing to avoid relying on src.data.transforms.Compose.
        Output shape: [1, 3, image_size, image_size]
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(
            img_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = torch.from_numpy(img_resized).float()
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def get_object_color(self, index):
        """
        Deterministic BGR colour per detected object.
        OpenCV uses BGR, not RGB.
        """
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

    def _normalize_keypoints_array(self, kps, image_w, image_h):
        """
        Converts keypoints into a list of visible pixel points.
        Supports:
        - [x1, y1, x2, y2, ...]
        - [x1, y1, score1, x2, y2, score2, ...]
        - [[x, y], [x, y], ...]
        - [[x, y, score], ...]
        """
        kps = np.asarray(kps)

        if kps.ndim == 1:
            if len(kps) % 3 == 0:
                kps = kps.reshape(-1, 3)[:, :2]
            else:
                kps = kps.reshape(-1, 2)
        elif kps.ndim == 2 and kps.shape[1] >= 2:
            kps = kps[:, :2]
        else:
            return []

        points = []

        for kp in kps:
            x, y = float(kp[0]), float(kp[1])

            # If normalized, scale to original image size.
            if x <= 1.5 and y <= 1.5:
                x *= image_w
                y *= image_h

            x, y = int(round(x)), int(round(y))

            if 0 <= x < image_w and 0 <= y < image_h:
                points.append((x, y))

        return points

    def draw_skeleton(self, img, kps, color):
        h, w = img.shape[:2]
        points = self._normalize_keypoints_array(kps, w, h)

        # Draw keypoints first.
        for x, y in points:
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.circle(img, (x, y), 7, (0, 0, 0), 1)

        # Draw simple chain skeleton.
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 2)

        return points

    def draw_label(self, img, text, anchor_point, color):
        x, y = anchor_point
        h, w = img.shape[:2]

        x = max(0, min(int(x), w - 1))
        y = max(20, min(int(y), h - 1))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        box_x1 = x
        box_y1 = y - text_h - baseline - 8
        box_x2 = x + text_w + 10
        box_y2 = y + baseline

        # Keep label inside image bounds.
        if box_x2 >= w:
            shift = box_x2 - w + 2
            box_x1 -= shift
            box_x2 -= shift

        if box_y1 < 0:
            box_y1 = y
            box_y2 = y + text_h + baseline + 8
            text_y = box_y1 + text_h + 4
        else:
            text_y = box_y2 - baseline - 4

        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(w - 1, box_x2)
        box_y2 = min(h - 1, box_y2)

        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color, -1)

        # White text is easier to read across most object colours.
        cv2.putText(
            img,
            text,
            (box_x1 + 5, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _get_label_anchor_from_keypoints(self, points, image_w, image_h):
        """
        Places label near the top-left of the object's visible keypoint cluster.
        """
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x = max(0, min(xs))
        y = max(20, min(ys) - 10)

        return x, y

    def infer_image(self, image_path, output_dir):
        img = cv2.imread(str(image_path))

        if img is None:
            print(f"Could not read image: {image_path}")
            return

        tensor = self.preprocess(img)

        with torch.no_grad():
            outputs = self.model(tensor)

            h, w = img.shape[:2]

            # PostProcess expects a tensor, not torch.Size.
            # Order is [width, height].
            target_sizes = torch.tensor(
                [[w, h]],
                dtype=torch.float32,
                device=self.device,
            )

            results = self.postprocessor(outputs, target_sizes)

        # Handle either list-style or dict-style postprocessor output.
        if isinstance(results, list):
            result = results[0]
        else:
            result = {
                "scores": results["scores"][0],
                "labels": results["labels"][0],
                "keypoints": results["keypoints"][0],
            }
            if "boxes" in results:
                result["boxes"] = results["boxes"][0]

        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        keypoints = result["keypoints"].detach().cpu().numpy()

        boxes = None
        if "boxes" in result:
            boxes = result["boxes"].detach().cpu().numpy()

        keep = scores > self.conf_thresh
        scores = scores[keep]
        labels = labels[keep]
        keypoints = keypoints[keep]

        if boxes is not None:
            boxes = boxes[keep]

        vis = img.copy()

        for i, (score, label, kps) in enumerate(zip(scores, labels, keypoints)):
            color = self.get_object_color(i)
            class_name = CLASS_NAMES.get(int(label), f"cls_{int(label)}")
            text = f"{class_name} {score:.3f}"

            # Draw keypoints/skeleton first in the object's colour.
            points = self.draw_skeleton(vis, kps, color)

            label_anchor = None

            # If the postprocessor returns boxes, draw the box and place label there.
            if boxes is not None:
                x1, y1, x2, y2 = boxes[i][:4]

                # If normalized, scale to original image size.
                if x2 <= 1.5 and y2 <= 1.5:
                    x1 *= w
                    x2 *= w
                    y1 *= h
                    y2 *= h

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label_anchor = (x1, y1)
            else:
                # Fallback when there are no boxes: place label next to the object's keypoint cluster.
                label_anchor = self._get_label_anchor_from_keypoints(points, w, h)

            if label_anchor is None:
                label_anchor = (10, 35 + i * 35)

            self.draw_label(vis, text, label_anchor, color)

        output_path = output_dir / f"pred_{image_path.name}"
        cv2.imwrite(str(output_path), vis)

        print(f"→ {image_path.name}: {len(scores)} detections")

    def infer_path(self, input_path, output_dir):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            self.infer_image(input_path, output_dir)
            return

        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(input_path.glob(ext))

        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"No images found in: {input_path}")
            return

        print(f"Found {len(image_paths)} images")

        for img_path in image_paths:
            self.infer_image(img_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--config",
        type=str,
        default="configs.detrpose.detrpose_hgnetv2_s_custom",
        help="Python module path to the training config",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use ckpt['model'] instead of EMA weights if EMA exists",
    )

    args = parser.parse_args()

    infer = DETRPoseInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf,
        config_module=args.config,
        use_ema=not args.no_ema,
        image_size=args.image_size,
    )

    infer.infer_path(args.input, args.output)

    print(f"\nDone. Results saved to: {args.output}")
