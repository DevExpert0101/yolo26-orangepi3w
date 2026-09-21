#!/usr/bin/env python3
"""Export YOLO26 as a 6-head ONNX for the Orange Pi Zero 3W NPU.

The A733 VIP9000 / ACUITY toolchain cannot run YOLO26 end-to-end NMS.
Radxa's official A733 zoo uses the same layout:

  box_p3 (1,4,80,80)  box_p4 (1,4,40,40)  box_p5 (1,4,20,20)
  cls_p3 (1,80,80,80) cls_p4 (1,80,40,40) cls_p5 (1,80,20,20)

Decode + NMS stay on CPU in infer.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn


class Yolo26SixHead(nn.Module):
    def __init__(self, detection_model: nn.Module):
        super().__init__()
        layers = detection_model.model
        self.layers = layers[:-1]
        self.detect = layers[-1]
        self.save = detection_model.save
        if not hasattr(self.detect, "cv2") or not hasattr(self.detect, "cv3"):
            raise RuntimeError("Not a YOLO Detect head. Use a detect .pt, not seg/pose/obb.")

    def forward(self, x: torch.Tensor):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        feats = [x if j == -1 else y[j] for j in self.detect.f]
        boxes, scores = [], []
        for i, feat in enumerate(feats):
            boxes.append(self.detect.cv2[i](feat))
            scores.append(self.detect.cv3[i](feat))
        return (*boxes, *scores)


YOLO26_SIZES = ("n", "s", "m", "l", "x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26 6-head ONNX for A733 NPU")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export yolo26n, yolo26s, yolo26m, yolo26l, and yolo26x",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--outdir", default="export")
    parser.add_argument("--calib", type=int, default=12, help="Calibration images to download")
    return parser.parse_args()


ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics", "onnx", "onnxslim"])


def _clean_partial_downloads(dest: Path) -> None:
    for pattern in (f".{dest.name}*.part", f"{dest.name}.download", f"{dest.name}.part"):
        for leftover in dest.parent.glob(pattern):
            try:
                leftover.unlink()
                print(f"Removed leftover {leftover.name}")
            except OSError as exc:
                print(f"Could not remove {leftover}: {exc}")


def ensure_local_weights(model: str) -> str:
    """Return a local .pt path. Avoids Ultralytics curl resume (HTTP 416 / WinError 32)."""
    path = Path(model)
    if path.is_file() and path.stat().st_size > 100_000:
        return str(path.resolve())

    name = path.name
    dest = path if path.is_absolute() else Path.cwd() / name
    if dest.is_file() and dest.stat().st_size > 100_000:
        return str(dest.resolve())

    if not name.endswith(".pt"):
        return model

    _clean_partial_downloads(dest)
    url = f"{ULTRALYTICS_ASSETS}/{name}"
    tmp = dest.with_name(dest.name + ".download")
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "orangePi-yolo26-export"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise ConnectionError(
            f"Could not download {url}: {exc}\n"
            f"Download it in a browser and pass --model path\\to\\{name}"
        ) from exc

    size = tmp.stat().st_size
    if size < 100_000:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ConnectionError(f"Downloaded {name} is too small ({size} bytes)")

    last_err: OSError | None = None
    for attempt in range(8):
        try:
            shutil.copyfile(tmp, dest)
            last_err = None
            break
        except OSError as exc:
            last_err = exc
            time.sleep(0.4 * (attempt + 1))
    if last_err is not None:
        raise PermissionError(
            f"Downloaded {tmp} but could not copy to {dest}: {last_err}. "
            f"Close other programs using that file, then: python export_npu.py --model {dest}"
        ) from last_err
    try:
        tmp.unlink()
    except OSError:
        pass
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    return str(dest.resolve())


def download_calib(calib_dir: Path, count: int) -> list[str]:
    calib_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://ultralytics.com/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg",
    ]
    saved: list[str] = []
    for i, url in enumerate(urls[:count]):
        dest = calib_dir / Path(url).name
        if not dest.exists():
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                print(f"calib download failed {url}: {exc}")
                continue
    saved.append(dest.name)
    if saved and count > len(saved):
        extra = calib_dir / "bus_repeat.jpg"
        extra.write_bytes((calib_dir / Path(saved[0]).name).read_bytes())
        while len(saved) < count:
            saved.append(extra.name)
    return saved[:count]


def write_acuity_sidecars(outdir: Path, stem: str, imgsz: int, nc: int, strides: list[int], calib: list[str]) -> None:
    (outdir / f"{stem}_inputmeta.yml").write_text(
        "\n".join(
            [
                f"# ACUITY input meta for {stem}",
                "# uint8 0-255 RGB, NCHW. Matches VIPLite data_format=2 on A733.",
                "inputs:",
                "  - name: images",
                f"    shape: [1, 3, {imgsz}, {imgsz}]",
                "    mean: [0.0, 0.0, 0.0]",
                "    scale: [255.0, 255.0, 255.0]",
                "    reverse_channel: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dataset = outdir / "dataset.txt"
    dataset.write_text("\n".join(calib) + ("\n" if calib else ""), encoding="utf-8")
    (outdir / f"{stem}.json").write_text(
        json.dumps(
            {
                "model": stem,
                "task": "detect",
                "imgsz": imgsz,
                "nc": nc,
                "strides": strides,
                "outputs": ["box_p3", "box_p4", "box_p5", "cls_p3", "cls_p4", "cls_p5"],
                "optimize": "VIP9000NANODI_PLUS_PID0X1000003B",
                "platform": "a733",
                "quant": "fp16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def export_one(model: str, args: argparse.Namespace, outdir: Path) -> Path:
    from ultralytics import YOLO

    stem = Path(model).name.replace(".pt", "") + "_6"
    onnx_path = outdir / f"{stem}.onnx"

    weights = ensure_local_weights(model)
    print(f"\n=== {Path(weights).name} → {onnx_path.name} ===")
    yolo = YOLO(weights)
    yolo.fuse()
    core = yolo.model.eval()
    detect = core.model[-1]
    nc = int(detect.nc)
    strides = [int(s) for s in detect.stride.tolist()]

    wrapper = Yolo26SixHead(core).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        outs = wrapper(dummy)
    print("6-head shapes:", [tuple(o.shape) for o in outs])

    print(f"Exporting {onnx_path} opset={args.opset}")
    export_kwargs = dict(
        input_names=["images"],
        output_names=["box_p3", "box_p4", "box_p5", "cls_p3", "cls_p4", "cls_p5"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kwargs)
    try:
        import onnx
        from onnxslim import slim

        slim(str(onnx_path), str(onnx_path))
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print("ONNX simplified")
    except Exception as exc:
        print(f"onnx simplify skipped: {exc}")
    try:
        from prepare_onnx_acuity import prepare as acuity_prepare

        acuity_prepare(onnx_path, backup=False)
    except Exception as exc:
        print(f"ACUITY ONNX rewrite skipped: {exc}")

    calib = download_calib(outdir / "calib", args.calib)
    dataset_lines = [f"calib/{name}" for name in calib]
    write_acuity_sidecars(outdir, stem, args.imgsz, nc, strides, dataset_lines)
    print(f"Wrote {onnx_path}")
    del yolo, wrapper, core
    return onnx_path


def main() -> int:
    args = parse_args()
    ensure_ultralytics()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    models = [f"yolo26{sz}.pt" for sz in YOLO26_SIZES] if args.all else [args.model]

    written: list[Path] = []
    failed: list[str] = []
    for model in models:
        try:
            written.append(export_one(model, args, outdir))
        except Exception as exc:
            print(f"FAILED {model}: {exc}")
            failed.append(f"{model}: {exc}")

    print("\nNext (Linux/WSL, native ACUITY — no Docker):")
    print("  export ACUITY_PATH=$HOME/acuity-toolkit-whl-6.30.22/bin")
    print("  export VIV_SDK=$HOME/Vivante_IDE/VivanteIDE5.11.0/cmdtools")
    for path in written:
        print(f"  ./convert_npu.sh --onnx {path} --quant fp16")
        print(f"  # INT8: ./convert_npu.sh --onnx {path} --quant pcq")
    if failed:
        print("Failed:")
        for line in failed:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
