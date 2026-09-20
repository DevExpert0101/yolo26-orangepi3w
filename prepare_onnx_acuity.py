#!/usr/bin/env python3
"""Rewrite YOLO26 ONNX so ACUITY pegasus can import it.

C2f/C3k2 export as Slice(starts/ends tensors). ACUITY 6.30 shape inference
then crashes in conv_shape (IndexError) on the following Conv.
Replace equal channel-split Slice pairs with Split.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _init_map(graph) -> dict[str, np.ndarray]:
    return {t.name: numpy_helper.to_array(t) for t in graph.initializer}


def _as_1d_int(arr: np.ndarray | None) -> list[int] | None:
    if arr is None:
        return None
    return [int(x) for x in np.asarray(arr).reshape(-1).tolist()]


def parse_channel_slice(node, inits: dict[str, np.ndarray]):
    if node.op_type != "Slice" or len(node.input) < 3:
        return None
    data, starts_n, ends_n = node.input[0], node.input[1], node.input[2]
    axes_n = node.input[3] if len(node.input) > 3 else ""
    steps_n = node.input[4] if len(node.input) > 4 else ""
    starts = _as_1d_int(inits.get(starts_n))
    ends = _as_1d_int(inits.get(ends_n))
    axes = _as_1d_int(inits.get(axes_n)) if axes_n else [0]
    steps = _as_1d_int(inits.get(steps_n)) if steps_n else [1]
    if not starts or not ends or axes is None or steps is None:
        return None
    if len(starts) != 1 or len(ends) != 1 or len(axes) != 1 or len(steps) != 1:
        return None
    if axes[0] != 1 or steps[0] != 1:
        return None
    return data, starts[0], ends[0], node.output[0]


def replace_c2f_slices(model: onnx.ModelProto) -> int:
    graph = model.graph
    inits = _init_map(graph)
    groups: dict[str, list] = defaultdict(list)
    for node in graph.node:
        parsed = parse_channel_slice(node, inits)
        if parsed is None:
            continue
        data, start, end, out = parsed
        groups[data].append((start, end, out, node.name))

    replace: dict[str, tuple] = {}
    for data, parts in groups.items():
        if len(parts) < 2:
            continue
        parts = sorted(parts, key=lambda p: p[0])
        sizes = []
        cursor = parts[0][0]
        if cursor != 0:
            continue
        ok = True
        outs = []
        names = []
        for start, end, out, name in parts:
            if start != cursor or end <= start:
                ok = False
                break
            sizes.append(end - start)
            outs.append(out)
            names.append(name)
            cursor = end
        if not ok or len(sizes) < 2:
            continue
        for name in names:
            replace[name] = (data, sizes, outs)

    if not replace:
        return 0

    used = set()
    new_nodes = []
    n_split = 0
    for node in graph.node:
        if node.name in replace and node.name not in used:
            data, sizes, outs = replace[node.name]
            for other, spec in replace.items():
                if spec[0] == data and spec[2] == outs:
                    used.add(other)
            split_name = f"{data.replace('/', '_')}_c2f_split"
            sizes_name = split_name + "_sizes"
            graph.initializer.append(
                numpy_helper.from_array(np.asarray(sizes, dtype=np.int64), name=sizes_name)
            )
            new_nodes.append(
                helper.make_node(
                    "Split",
                    inputs=[data, sizes_name],
                    outputs=list(outs),
                    axis=1,
                    name=split_name,
                )
            )
            n_split += 1
            continue
        if node.name in used:
            continue
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return n_split


def drop_empty_resize_roi(model: onnx.ModelProto) -> int:
    """Resize with an empty roi input ('') confuses some ACUITY scanners."""
    n = 0
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        ins = list(node.input)
        if len(ins) >= 2 and ins[1] == "":
            # X, roi='', scales=...  →  keep X and scales; insert empty roi tensor
            roi_name = node.name.replace("/", "_") + "_roi_empty"
            model.graph.initializer.append(
                numpy_helper.from_array(np.zeros((0,), dtype=np.float32), name=roi_name)
            )
            ins[1] = roi_name
            del node.input[:]
            node.input.extend(ins)
            n += 1
    return n


def prepare(path: Path, backup: bool = True) -> Path:
    model = onnx.load(str(path))
    n_split = replace_c2f_slices(model)
    n_roi = drop_empty_resize_roi(model)
    print(f"{path.name}: Slice->Split {n_split}, Resize roi {n_roi}")
    if n_split == 0 and n_roi == 0:
        ops = {n.op_type for n in model.graph.node}
        if "Slice" not in ops:
            print("Already ACUITY-friendly (no Slice).")
            return path
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"shape_inference warning: {exc}")
    if backup:
        bak = path.with_suffix(path.suffix + ".preacuity")
        if not bak.exists():
            shutil.copyfile(path, bak)
            print(f"Backup {bak.name}")
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("ops", ops)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    path = Path(args.onnx)
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1
    prepare(path, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
