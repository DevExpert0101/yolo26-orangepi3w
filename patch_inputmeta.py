#!/usr/bin/env python3
"""Fill ACUITY inputmeta: TEXT dataset + YOLO 0-255 -> /255 scale.

Must run with the image Python 3.8 (acuitylib). CWD = export dir.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolo26n_6"
    dataset = Path("dataset.txt")
    model = Path(f"{name}.json")
    meta = Path(f"{name}_inputmeta.yml")
    if not model.is_file():
        print("missing", model)
        return 1

    lines = []
    if dataset.is_file():
        for raw in dataset.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_file():
                # Windows path from export_npu.py → relative calib/
                p = Path("calib") / Path(line.replace("\\", "/")).name
            if p.is_file():
                lines.append(p.as_posix())
    if not lines:
        print("No calibration images. Put JPEGs in calib/ and list them in dataset.txt")
        return 1
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("dataset.txt ->", lines)

    from acuitylib.vsi_nn import VSInn

    nn = VSInn()
    net = nn.create_net()
    nn.load_model(net, str(model))
    if meta.is_file():
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
    port.preprocess["mean"] = [0.0] * channel
    port.preprocess["scale"] = [1.0 / 255.0] * channel
    port.preprocess["reverse_channel"] = False
    net.update_input_meta(data)
    nn.save_model_inputmeta(net, str(meta))
    print("wrote", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
