"""
RTMOPose ONNX Export Tool
Exports a trained RTMOPose checkpoint to an ONNX model.

Usage:
    python tools/deployment/export_rtmo_onnx.py \
        --config  configs/rtmopose/rtmopose_hgnetv2_s_custom.py \
        --resume  output/rtmopose_hgnetv2_s_custom/checkpoint_best_regular.pth \
        --check --simplify

The exported ONNX model has:
    Inputs:  images [N, 3, 640, 640], orig_target_sizes [N, 2]
    Outputs: scores, labels, keypoints, boxes  (same as DETRPose ONNX)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import LazyConfig, instantiate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../inference"))
try:
    from class_mapping_utils import save_class_mappings_json  # type: ignore
except ImportError:
    def save_class_mappings_json(mappings, path, **kwargs):
        import json
        with open(path, "w") as f:
            json.dump({"class_mappings": mappings, **kwargs}, f, indent=2)


# ── Exportable wrapper ────────────────────────────────────────────────────────

class RTMOExportWrapper(nn.Module):
    """
    Thin wrapper that runs the model in eval mode and returns
    flat tensors for ONNX export:
        scores, labels, keypoints, boxes
    """

    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, orig_target_sizes):
        """
        images:            [N, 3, H, W]
        orig_target_sizes: [N, 2]  (H, W per image)

        Returns (scores, labels, keypoints, boxes) – each [N, max_det] or [N, max_det, ...]
        Padded to max_det = 300.
        """
        outputs = self.model(images)
        results = self.postprocessor(outputs, orig_target_sizes)

        # Stack into batch tensors (padded to max_detections)
        max_det = self.postprocessor.max_detections
        B = len(results)
        device = images.device

        scores_out    = torch.zeros(B, max_det, device=device)
        labels_out    = torch.zeros(B, max_det, dtype=torch.int64, device=device)
        boxes_out     = torch.zeros(B, max_det, 4, device=device)
        K = results[0]["keypoints"].shape[-1]
        keypoints_out = torch.zeros(B, max_det, K, device=device)

        for i, r in enumerate(results):
            n = min(r["scores"].shape[0], max_det)
            scores_out[i,    :n]    = r["scores"][:n]
            labels_out[i,    :n]    = r["labels"][:n]
            boxes_out[i,     :n]    = r["boxes"][:n]
            keypoints_out[i, :n]    = r["keypoints"][:n]

        return scores_out, labels_out, keypoints_out, boxes_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    cfg = LazyConfig.load(args.config_file)

    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    postprocessor.deploy_mode = True

    class_mappings     = {}
    skeleton_connections = {}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

        if "class_mappings" in checkpoint:
            class_mappings = checkpoint["class_mappings"]
            print(f"Loaded class mappings: {class_mappings}")

        if "skeleton_connections" in checkpoint:
            raw_sk = checkpoint["skeleton_connections"]
            skeleton_connections = {
                str(k): [list(pair) for pair in v]
                for k, v in raw_sk.items()
            }

        if "ema" in checkpoint and checkpoint["ema"] is not None:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Strip torch.compile prefix
        state = {
            (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state)
    else:
        print("Warning: no checkpoint provided; exporting with random weights.")

    model.eval()

    export_model = RTMOExportWrapper(model, postprocessor)
    export_model.eval()

    # Warm-up pass
    dummy_img  = torch.rand(1, 3, 640, 640)
    dummy_size = torch.tensor([[640, 640]])
    with torch.no_grad():
        _ = export_model(dummy_img, dummy_size)

    os.makedirs("onnx_engines", exist_ok=True)
    cfg_name = os.path.basename(args.config_file).replace(".py", "")
    output_file = f"onnx_engines/{cfg_name}.onnx"

    dynamic_axes = {
        "images":            {0: "N"},
        "orig_target_sizes": {0: "N"},
    }

    torch.onnx.export(
        export_model,
        (dummy_img, dummy_size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["scores", "labels", "keypoints", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to: {output_file}")

    if class_mappings or skeleton_connections:
        json_path = output_file.replace(".onnx", "_class_mappings.json")
        save_class_mappings_json(class_mappings, json_path, skeleton_connections=skeleton_connections)
        print(f"Saved class mappings to: {json_path}")

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    if args.simplify:
        import onnx
        import onnxsim
        input_shapes = {"images": [1, 3, 640, 640], "orig_target_sizes": [1, 2]}
        model_sim, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(model_sim, output_file)
        print(f"ONNX simplification: {check}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export RTMOPose to ONNX")
    parser.add_argument("--config_file", "-c",
                        default="configs/rtmopose/rtmopose_hgnetv2_s_custom.py")
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--check",    action="store_true", default=True)
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
