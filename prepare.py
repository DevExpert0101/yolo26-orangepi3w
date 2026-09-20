#!/usr/bin/env python3
"""Prepare YOLO26 models for Orange Pi Zero 3W (Allwinner A733).

Run this on a PC (Windows/Linux/macOS) with enough RAM. Copy the exported
folder to the board afterwards. Exporting on the board is possible only if
it has roughly 4 GB+ free RAM.

Exports:
  - npu    : 6-head ONNX for A733 VIP9000 (then convert_npu.sh -> .nb)
  - NCNN   : ARM CPU fallback
  - ONNX   : ARM CPU fallback
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


MODELS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26 for Orange Pi 3W")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics weights. Prefer yolo26n.pt on 1-4 GB boards.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export input size")
    parser.add_argument(
        "--formats",
        default="npu",
        help="Comma list: npu,ncnn,onnx",
    )
    parser.add_argument(
        "--outdir",
        default="export",
        help="Directory that will be copied to the Orange Pi",
    )
    parser.add_argument(
        "--quantize",
        default="16",
        help="NCNN weight precision: 16 (FP16, recommended) or 32",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset. 12 is safer for older NPU toolchains.",
    )
    return parser.parse_args()


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("Installing ultralytics and export extras...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "ultralytics", "onnx", "onnxslim"]
        )


def model_stem(model: str) -> str:
    name = Path(model).name
    return name[:-3] if name.endswith(".pt") else Path(model).stem


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Export output not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def export_one(model, fmt: str, imgsz: int, quantize: str, opset: int, outdir: Path, stem: str, model_path: str) -> Path:
    if fmt == "ncnn":
        exported = Path(
            model.export(format="ncnn", imgsz=imgsz, quantize=int(quantize), device="cpu")
        )
        dest = outdir / f"{stem}_ncnn_model"
        copy_if_exists(exported, dest)
        return dest

    if fmt == "onnx":
        exported = Path(
            model.export(
                format="onnx",
                imgsz=imgsz,
                simplify=True,
                opset=opset,
                device="cpu",
            )
        )
        dest = outdir / exported.name
        copy_if_exists(exported, dest)
        return dest

    if fmt in {"npu", "npu-onnx"}:
        import subprocess

        script = Path(__file__).resolve().parent / "export_npu.py"
        subprocess.check_call(
            [
                sys.executable,
                str(script),
                "--model",
                model_path,
                "--imgsz",
                str(imgsz),
                "--opset",
                str(max(opset, 16)),
                "--outdir",
                str(outdir),
            ]
        )
        dest = outdir / f"{stem}_6.onnx"
        if not dest.exists():
            matches = list(outdir.glob("*_6.onnx"))
            if not matches:
                raise FileNotFoundError("6-head NPU ONNX was not created")
            dest = matches[0]
        return dest

    raise ValueError(f"Unknown format: {fmt}")


def write_manifest(outdir: Path, args: argparse.Namespace, artifacts: list[Path]) -> None:
    lines = [
        "YOLO26 Orange Pi Zero 3W export",
        f"model={args.model}",
        f"imgsz={args.imgsz}",
        f"quantize={args.quantize}",
        f"opset={args.opset}",
        "",
        "NPU next steps (Linux/WSL, native ACUITY, no Docker):",
        "  1. export ACUITY_PATH and VIV_SDK, then ./convert_npu.sh --onnx export/yolo26n_6.onnx",
        "  2. copy the .nb to the Orange Pi",
        "  3. python infer.py --model export/*.nb --source export/bus.jpg",
        "",
        "Artifacts:",
    ]
    for path in artifacts:
        lines.append(f"  {path.relative_to(outdir.parent)}")
    (outdir / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    stem = model_stem(args.model)
    if stem.split(".")[0] not in MODELS and not Path(args.model).exists():
        print(f"Warning: {args.model} is not a stock YOLO26 name. Continuing anyway.")

    ensure_ultralytics()
    from ultralytics import YOLO

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)

    artifacts: list[Path] = []
    for fmt in [item.strip().lower() for item in args.formats.split(",") if item.strip()]:
        print(f"\n=== Export {fmt} imgsz={args.imgsz} ===")
        dest = export_one(model, fmt, args.imgsz, args.quantize, args.opset, outdir, stem, args.model)
        print(f"Wrote {dest}")
        artifacts.append(dest)

    sample = outdir / "bus.jpg"
    if not sample.exists():
        try:
            import urllib.request

            urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", sample)
            artifacts.append(sample)
        except Exception as exc:
            print(f"Could not download sample image: {exc}")

    write_manifest(outdir, args, artifacts)
    print(f"\nDone. Copy the '{outdir.name}' folder to the Orange Pi Zero 3W.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
