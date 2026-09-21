#!/usr/bin/env python3
"""Export a standard YOLO26 ONNX for Orange Pi Zero 3W (onnxruntime CPU).

Run this on a PC. Do not export on a 1 GB board.

This is the end-to-end detect graph (boxes + scores + class), not the
6-head NPU ONNX from export_npu.py. Copy export_onnx/ to the board and
run Yolo26_ONNX (C++) or infer.py --backend onnx.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26 ONNX for Orange Pi 3W CPU")
    parser.add_argument("--model", default="yolo26n.pt", help="Prefer yolo26n.pt on 1-2 GB boards")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12, help="12 is safest for older onnxruntime")
    parser.add_argument("--outdir", default="export_onnx")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Export one-to-many (1,84,8400) instead of YOLO26 e2e (1,300,6)",
    )
    return parser.parse_args()


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "-r", str(Path(__file__).with_name("requirements-prepare.txt"))]
        )


def write_manifest(outdir: Path, model: str, dest: Path, imgsz: int, opset: int) -> None:
    lines = [
        "YOLO26 ONNX for Orange Pi Zero 3W (CPU / onnxruntime)",
        f"model={model}",
        f"onnx={dest.name}",
        f"imgsz={imgsz}",
        f"opset={opset}",
        "",
        "On the board:",
        "  ./setup_onnx.sh",
        "  cd Yolo26_ONNX && ./build.sh",
        "  ./run_onnx.sh",
        "  # Python optional:",
        "  #   ./setup_onnx.sh --python && source .venv/bin/activate",
        f"  #   python infer.py --backend onnx --model {outdir.name}/{dest.name} --source {outdir.name}/bus.jpg",
        "",
        "Do not pass this file to convert_npu.sh. NPU needs export_npu.py (*_6.onnx).",
    ]
    (outdir / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_ultralytics()
    from ultralytics import YOLO

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)
    export_kw = dict(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=args.opset,
        device="cpu",
    )
    # YOLO26: nms=False is the native one-to-one head (1,300,6). nms=True embeds
    # older NMS ops that many aarch64 ORT builds reject.
    if args.raw:
        export_kw["nms"] = None
        export_kw["end2end"] = False
    else:
        export_kw["nms"] = False

    print(f"Exporting ONNX imgsz={args.imgsz} opset={args.opset} e2e={not args.raw}")
    exported = Path(model.export(**export_kw))
    dest = outdir / exported.name
    if exported.resolve() != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, dest)

    sample = outdir / "bus.jpg"
    if not sample.exists():
        try:
            urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", sample)
            print(f"Wrote {sample}")
        except Exception as exc:
            print(f"Could not download sample image: {exc}")

    write_manifest(outdir, args.model, dest, args.imgsz, args.opset)
    print(f"Wrote {dest}")
    print(f"\nCopy the '{outdir.name}' folder to the Orange Pi Zero 3W.")
    print("On the board:  ./setup_onnx.sh && cd Yolo26_ONNX && ./build.sh && cd .. && ./run_onnx.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
