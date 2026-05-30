"""
RTMOPose PyTorch Inference Tool
Runs inference using a trained RTMOPose checkpoint.

Usage:
    python tools/inference/rtmo_inf.py \
        --checkpoint output/rtmopose_hgnetv2_s_custom/checkpoint_best_regular.pth \
        --config   configs.rtmopose.rtmopose_hgnetv2_s_custom \
        --source   data/coco/val/images \
        --device   cuda
"""

import argparse
import importlib
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core import instantiate
except ImportError:
    from src.core.workspace import instantiate  # type: ignore


class RTMOInference:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        conf_thresh=0.35,
        config_module="configs.rtmopose.rtmopose_hgnetv2_s_custom",
        use_ema=True,
        image_size=640,
    ):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.conf_thresh = conf_thresh
        self.image_size = image_size

        print(f"Using device: {self.device}")
        print(f"Loading config: {config_module}")

        cfg = importlib.import_module(config_module)

        self.model = instantiate(cfg.model)
        self.postprocessor = instantiate(cfg.postprocessor)

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.class_mappings = ckpt.get("class_mappings", {})
        if self.class_mappings:
            print("Loaded class mappings from checkpoint:")
            for cid, name in sorted(self.class_mappings.items()):
                print(f"  ID {cid}: {name}")
        else:
            print("Warning: No class mappings in checkpoint. Using numeric IDs.")

        self.skeleton_connections = ckpt.get("skeleton_connections", {})

        state_dict = self._extract_state_dict(ckpt, use_ema=use_ema)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict = {
            (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_state_dict(self, ckpt, use_ema=True):
        if use_ema and "ema" in ckpt and ckpt["ema"] is not None:
            print("Using EMA weights")
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            print("Using model weights")
            return ckpt["model"]
        return ckpt

    # ─────────────────────────────────────────────────────────────────────────
    def preprocess(self, image_bgr):
        """BGR uint8 → [1, 3, H, W] float32 tensor (values 0-255)."""
        h, w = image_bgr.shape[:2]
        scale = self.image_size / max(h, w)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        resized = cv2.resize(image_bgr, (new_w, new_h))

        padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        padded[:new_h, :new_w] = resized.astype(np.float32)

        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor.to(self.device), scale, (new_w, new_h)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def infer(self, image_bgr):
        """
        Run inference on a single BGR image.

        Returns list of dicts:
            {'score', 'label', 'label_name', 'box' (xyxy), 'keypoints' (K, 3)}
        """
        tensor, scale, (pad_w, pad_h) = self.preprocess(image_bgr)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            outputs = self.model(tensor)

        if isinstance(outputs, list):
            results_raw = outputs[0]
        else:
            # Training mode fallback — run postprocessor manually
            h_net = w_net = self.image_size
            orig_sizes = torch.tensor([[h_net, w_net]], device=self.device)
            results_raw = self.postprocessor(outputs, orig_sizes)[0]

        detections = []
        for i in range(len(results_raw["scores"])):
            score = results_raw["scores"][i].item()
            if score < self.conf_thresh:
                continue

            label = results_raw["labels"][i].item()
            box   = results_raw["boxes"][i].cpu().numpy() / scale
            kpts  = results_raw["keypoints"][i].cpu().numpy()

            K = len(kpts) // 3
            kpts_arr = kpts.reshape(K, 3)
            kpts_arr[:, :2] /= scale   # scale coords back to original image

            detections.append({
                "score":      score,
                "label":      label,
                "label_name": self.class_mappings.get(label, str(label)),
                "box":        box,
                "keypoints":  kpts_arr,
            })

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    def draw(self, image_bgr, detections, show_skeleton=True):
        """Annotate and return an image."""
        img = image_bgr.copy()
        palette = np.random.RandomState(42).randint(50, 220, (256, 3)).tolist()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cid = det["label"] % 256
            color = palette[cid]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{det['label_name']} {det['score']:.2f}",
                        (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            kpts = det["keypoints"]
            K = len(kpts)
            for k in range(K):
                x, y, vis = kpts[k]
                if vis > 0.3:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

            if show_skeleton:
                skel = self.skeleton_connections.get(
                    det["label"],
                    self.skeleton_connections.get(str(det["label"]), [])
                )
                for a, b in skel:
                    if a < K and b < K and kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                        cv2.line(img,
                                 (int(kpts[a, 0]), int(kpts[a, 1])),
                                 (int(kpts[b, 0]), int(kpts[b, 1])),
                                 (255, 200, 0), 2)
        return img


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="RTMOPose inference")
    p.add_argument("--checkpoint", "-r", required=True)
    p.add_argument("--config", "-c",
                   default="configs.rtmopose.rtmopose_hgnetv2_s_custom")
    p.add_argument("--source", "-s", required=True,
                   help="Image file, directory, or glob pattern")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf",   type=float, default=0.35)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--output", "-o", default="predictions")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--show", action="store_true",
                   help="Display results with cv2.imshow (needs display)")
    return p.parse_args()


def _iter_images(source):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(source)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in exts:
                yield f
    else:
        import glob
        for f in sorted(glob.glob(source)):
            yield Path(f)


def main():
    args = _parse_args()
    infer = RTMOInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf,
        config_module=args.config,
        use_ema=not args.no_ema,
        image_size=args.image_size,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in _iter_images(args.source):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Could not read: {img_path}")
            continue

        detections = infer.infer(img_bgr)
        print(f"{img_path.name}: {len(detections)} detections")

        annotated = infer.draw(img_bgr, detections)
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        if args.show:
            cv2.imshow("RTMOPose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
