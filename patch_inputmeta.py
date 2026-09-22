#!/usr/bin/env python3
"""Match Allwinner model-zoo yolo11 convert_model/config_yml.py.

ACUITY YOLO graphs on A733 use:
  mean 0, scale 1/255, RGB
  add_preproc_node True, IMAGE_RGB   → NBG input is HWC uint8 0-255
  add_postproc_node True            → NBG outputs are float32

Must run with the image Python 3.8 (acuitylib). CWD = export dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Same numbers as linux_aw_npu/model_zoo/.../yolo11/convert_model/config_yml.py
MEAN = [0.0, 0.0, 0.0]
SCALE = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
REVERSE_CHANNEL = False
ADD_PREPROC_NODE = True
PREPROC_TYPE = "IMAGE_RGB"
ADD_POSTPROC_NODE = True


def rewrite_dataset(dataset: Path) -> list[str]:
    lines: list[str] = []
    if dataset.is_file():
        for raw in dataset.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_file():
                p = Path("calib") / Path(line.replace("\\", "/")).name
            if p.is_file():
                lines.append(p.as_posix())
    if not lines:
        print("No calibration images. Put JPEGs in calib/ and list them in dataset.txt")
        return []
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("dataset.txt ->", lines)
    return lines


def enable_postproc(path: Path) -> None:
    if not path.is_file():
        print("missing", path.name, "(run: pegasus generate postprocess-file)")
        return
    text = path.read_text(encoding="utf-8")
    updated = (
        text.replace("add_postproc_node: false", "add_postproc_node: true").replace(
            "add_postproc_node: False", "add_postproc_node: true"
        )
    )
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print("add_postproc_node: false -> true in", path.name)
    elif "add_postproc_node: true" in updated:
        print(path.name, "already has add_postproc_node: true")
    else:
        print("warning:", path.name, "has no add_postproc_node keys")


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n_6"
    dataset = Path("dataset.txt")
    model = Path(f"{name}.json")
    meta = Path(f"{name}_inputmeta.yml")
    post = Path(f"{name}_postprocess_file.yml")
    if not model.is_file():
        print("missing", model)
        return 1
    if not rewrite_dataset(dataset):
        return 1
    if not meta.is_file():
        print("missing", meta, "(run: pegasus generate inputmeta)")
        return 1

    from acuitylib.vsi_nn import VSInn

    nn = VSInn()
    net = nn.create_net()
    nn.load_model(net, str(model))
    try:
        nn.load_model_inputmeta(net, str(meta))
    except Exception as exc:
        print("existing inputmeta ignored:", exc)

    nn.set_database(net, dataset_files=str(dataset.resolve()), dataset_type="TEXT")
    data = net.get_input_meta()
    port = data.databases[0].ports[0]
    channel = 3
    try:
        if len(port.shape) == 4:
            channel = int(port.shape[1] if getattr(port, "layout", "nchw") == "nchw" else port.shape[-1])
    except Exception:
        pass
    if channel not in (1, 3, 4):
        channel = 3

    port.preprocess["mean"] = MEAN[:channel]
    scale = port.preprocess.get("scale")
    if isinstance(scale, (int, float)):
        port.preprocess["scale"] = SCALE[0]
    elif hasattr(scale, "__len__") and len(scale) == 1:
        port.preprocess["scale"] = SCALE[0]
    else:
        port.preprocess["scale"] = SCALE[:channel]
    port.preprocess["reverse_channel"] = REVERSE_CHANNEL

    params = port.preprocess["preproc_node_params"]
    params["add_preproc_node"] = ADD_PREPROC_NODE
    params["preproc_type"] = PREPROC_TYPE

    net.update_input_meta(data)
    nn.save_model_inputmeta(net, str(meta))
    print(
        "wrote",
        meta,
        "mean=0 scale=1/255 RGB add_preproc_node=",
        ADD_PREPROC_NODE,
        PREPROC_TYPE,
    )

    text = meta.read_text(encoding="utf-8")
    if "add_preproc_node: false" in text or "add_preproc_node: False" in text:
        text = text.replace("add_preproc_node: false", "add_preproc_node: true").replace(
            "add_preproc_node: False", "add_preproc_node: true"
        )
        if "preproc_type: TENSOR" in text:
            text = text.replace("preproc_type: TENSOR", f"preproc_type: {PREPROC_TYPE}")
        meta.write_text(text, encoding="utf-8")
        print("forced add_preproc_node: true /", PREPROC_TYPE, "in", meta.name)
        text = meta.read_text(encoding="utf-8")
    if "add_preproc_node: true" not in text.lower():
        print("ERROR:", meta, "does not contain add_preproc_node: true")
        return 1
    if "add_preproc_node: false" in text.lower():
        print("ERROR:", meta, "still has add_preproc_node: false")
        return 1

    if ADD_POSTPROC_NODE:
        enable_postproc(post)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
