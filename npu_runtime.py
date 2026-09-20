#!/usr/bin/env python3
"""VIPLite 2.x runtime for Allwinner A733 / Orange Pi Zero 3W.

Loads libNBGlinker.so + libVIPhal.so and runs an ACUITY NBG (*.nb) on
/dev/vipcore. YOLO26 decode (dist2bbox + NMS) stays on CPU.

If a VIPLite symbol layout does not match this board image, run:
  python npu_runtime.py --probe
and check `nm -D $(find /usr -name libNBGlinker.so | head -1) | grep vip_`
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    c_char_p,
    c_float,
    c_int32,
    c_uint32,
    c_void_p,
    c_size_t,
)
from pathlib import Path

import numpy as np


VIP_SUCCESS = 0
VIP_MAX_DIM = 6
VIP_CREATE_NETWORK_FROM_MEMORY = 0
VIP_CREATE_NETWORK_FROM_FILE = 1
VIP_BUFFER_NULL = 0
VIP_BUFFER_SYNC_FOR_READ = 1
VIP_BUFFER_SYNC_FOR_WRITE = 2

VIP_NETWORK_PROP_LAYER_COUNT = 0
VIP_NETWORK_PROP_INPUT_COUNT = 1
VIP_NETWORK_PROP_OUTPUT_COUNT = 2

VIP_BUFFER_PROP_DATA_FORMAT = 0
VIP_BUFFER_PROP_NUM_OF_DIMENSION = 1
VIP_BUFFER_PROP_SIZES_OF_DIMENSION = 2
VIP_BUFFER_PROP_QUANT_FORMAT = 3
VIP_BUFFER_PROP_QUANT_DATA = 4
VIP_BUFFER_PROP_NAME = 5

VIP_BUFFER_FORMAT_FP32 = 0
VIP_BUFFER_FORMAT_FP16 = 1
VIP_BUFFER_FORMAT_UINT8 = 2
VIP_BUFFER_FORMAT_INT8 = 3
VIP_BUFFER_FORMAT_INT16 = 4

FORMAT_DTYPE = {
    VIP_BUFFER_FORMAT_FP32: np.float32,
    VIP_BUFFER_FORMAT_FP16: np.float16,
    VIP_BUFFER_FORMAT_UINT8: np.uint8,
    VIP_BUFFER_FORMAT_INT8: np.int8,
    VIP_BUFFER_FORMAT_INT16: np.int16,
}


class VipBufferCreateParams(ctypes.Structure):
    _fields_ = [
        ("num_of_dims", c_uint32),
        ("sizes", c_uint32 * VIP_MAX_DIM),
        ("data_format", c_int32),
        ("quant_format", c_int32),
        ("quant_scale", c_float),
        ("quant_zero_point", c_int32),
        ("memory_type", c_uint32),
    ]


def _find_lib(name: str) -> str:
    env = os.environ.get("VIPLITE_LIB_DIR", "")
    roots = [
        env,
        "/usr/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/local/lib",
        "/opt/yolov5",
        "/opt/npu",
        str(Path(__file__).resolve().parent / "lib"),
    ]
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / name
        if candidate.exists():
            return str(candidate)
        for hit in Path(root).rglob(name) if Path(root).is_dir() else []:
            return str(hit)
    raise FileNotFoundError(
        f"{name} not found. Install the Orange Pi / Allwinner NPU userspace "
        "and/or set VIPLITE_LIB_DIR to the folder with libNBGlinker.so."
    )


def _ok(status: int, what: str) -> None:
    if status != VIP_SUCCESS:
        raise RuntimeError(f"{what} failed, vip_status={status}")


class VipLite:
    def __init__(self, model: str, malloc_mb: int = 16):
        if not Path("/dev/vipcore").exists():
            raise SystemExit("NPU node /dev/vipcore is missing. Use the vendor kernel image.")
        self.nbg = Path(model)
        if not self.nbg.exists():
            raise FileNotFoundError(model)

        linker = _find_lib("libNBGlinker.so")
        try:
            ctypes.CDLL(_find_lib("libVIPhal.so"), mode=ctypes.RTLD_GLOBAL)
        except FileNotFoundError:
            pass
        self.lib = ctypes.CDLL(linker, mode=ctypes.RTLD_GLOBAL)
        self._bind()
        self._init(malloc_mb)
        self.net = c_void_p()
        path = str(self.nbg).encode("utf-8")
        _ok(
            self.lib.vip_create_network(path, 0, VIP_CREATE_NETWORK_FROM_FILE, ctypes.byref(self.net)),
            "vip_create_network",
        )
        _ok(self.lib.vip_prepare_network(self.net), "vip_prepare_network")
        self.inputs = self._query_io(is_input=True)
        self.outputs = self._query_io(is_input=False)
        print("NPU inputs:", self.inputs)
        print("NPU outputs:", self.outputs)
        self.in_bufs = [self._make_buffer(info, i, True) for i, info in enumerate(self.inputs)]
        self.out_bufs = [self._make_buffer(info, i, False) for i, info in enumerate(self.outputs)]

    def _bind(self) -> None:
        L = self.lib
        L.vip_create_network.argtypes = [c_void_p, c_size_t, c_int32, POINTER(c_void_p)]
        L.vip_create_network.restype = c_int32
        L.vip_prepare_network.argtypes = [c_void_p]
        L.vip_prepare_network.restype = c_int32
        L.vip_query_network.argtypes = [c_void_p, c_int32, c_void_p]
        L.vip_query_network.restype = c_int32
        L.vip_query_input.argtypes = [c_void_p, c_uint32, c_int32, c_void_p]
        L.vip_query_input.restype = c_int32
        L.vip_query_output.argtypes = [c_void_p, c_uint32, c_int32, c_void_p]
        L.vip_query_output.restype = c_int32
        L.vip_create_buffer.argtypes = [POINTER(VipBufferCreateParams), c_uint32, POINTER(c_void_p)]
        L.vip_create_buffer.restype = c_int32
        L.vip_set_input.argtypes = [c_void_p, c_uint32, c_void_p]
        L.vip_set_input.restype = c_int32
        L.vip_set_output.argtypes = [c_void_p, c_uint32, c_void_p]
        L.vip_set_output.restype = c_int32
        L.vip_run_network.argtypes = [c_void_p]
        L.vip_run_network.restype = c_int32
        L.vip_flush_buffer.argtypes = [c_void_p, c_int32]
        L.vip_flush_buffer.restype = c_int32
        L.vip_map_buffer.argtypes = [c_void_p]
        L.vip_map_buffer.restype = c_void_p
        L.vip_unmap_buffer.argtypes = [c_void_p]
        L.vip_unmap_buffer.restype = c_int32
        L.vip_destroy_buffer.argtypes = [c_void_p]
        L.vip_destroy_buffer.restype = c_int32
        L.vip_finish_network.argtypes = [c_void_p]
        L.vip_finish_network.restype = c_int32
        L.vip_destroy_network.argtypes = [c_void_p]
        L.vip_destroy_network.restype = c_int32
        L.vip_destroy.argtypes = []
        L.vip_destroy.restype = c_int32

    def _init(self, malloc_mb: int) -> None:
        errors = []
        for args in [(), (c_uint32(0),), (c_uint32(malloc_mb),), (c_uint32(malloc_mb * 1024 * 1024),)]:
            try:
                self.lib.vip_init.argtypes = [type(a) for a in args]
                self.lib.vip_init.restype = c_int32
                status = self.lib.vip_init(*args)
                if status == VIP_SUCCESS:
                    return
                errors.append(f"vip_init{args} -> {status}")
            except Exception as exc:
                errors.append(str(exc))
        raise RuntimeError("vip_init failed:\n  " + "\n  ".join(errors))

    def _q(self, fn, index: int, prop: int, buf) -> None:
        _ok(fn(self.net, index, prop, ctypes.byref(buf) if not isinstance(buf, ctypes.Array) else buf), f"query {prop}")

    def _query_io(self, is_input: bool) -> list[dict]:
        count = c_uint32(0)
        prop = VIP_NETWORK_PROP_INPUT_COUNT if is_input else VIP_NETWORK_PROP_OUTPUT_COUNT
        _ok(self.lib.vip_query_network(self.net, prop, ctypes.byref(count)), "query io count")
        query = self.lib.vip_query_input if is_input else self.lib.vip_query_output
        items = []
        for i in range(count.value):
            dims_n = c_uint32(0)
            fmt = c_int32(0)
            sizes = (c_uint32 * VIP_MAX_DIM)()
            name = ctypes.create_string_buffer(256)
            query(self.net, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION, ctypes.byref(dims_n))
            query(self.net, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, sizes)
            query(self.net, i, VIP_BUFFER_PROP_DATA_FORMAT, ctypes.byref(fmt))
            query(self.net, i, VIP_BUFFER_PROP_NAME, name)
            shape = [sizes[j] for j in range(dims_n.value)]
            items.append({"index": i, "name": name.value.decode("utf-8", "ignore"), "shape": shape, "format": fmt.value})
        return items

    def _make_buffer(self, info: dict, index: int, is_input: bool) -> c_void_p:
        params = VipBufferCreateParams()
        params.num_of_dims = len(info["shape"])
        for i, s in enumerate(info["shape"]):
            params.sizes[i] = s
        params.data_format = info["format"]
        params.quant_format = 0
        params.quant_scale = 1.0
        params.quant_zero_point = 0
        params.memory_type = 0
        handle = c_void_p()
        _ok(self.lib.vip_create_buffer(ctypes.byref(params), ctypes.sizeof(params), ctypes.byref(handle)), "create_buffer")
        setter = self.lib.vip_set_input if is_input else self.lib.vip_set_output
        _ok(setter(self.net, index, handle), "set io")
        return handle

    def infer(self, rgb_uint8: np.ndarray) -> list[np.ndarray]:
        if rgb_uint8.ndim != 3:
            raise ValueError("expected HxWx3 uint8 RGB")
        packed = self._pack_input(rgb_uint8)
        in_ptr = self.lib.vip_map_buffer(self.in_bufs[0])
        ctypes.memmove(in_ptr, packed.ctypes.data, packed.nbytes)
        self.lib.vip_unmap_buffer(self.in_bufs[0])
        _ok(self.lib.vip_flush_buffer(self.in_bufs[0], VIP_BUFFER_SYNC_FOR_WRITE), "flush input")
        _ok(self.lib.vip_run_network(self.net), "vip_run_network")
        outs = []
        for info, buf in zip(self.outputs, self.out_bufs):
            _ok(self.lib.vip_flush_buffer(buf, VIP_BUFFER_SYNC_FOR_READ), "flush output")
            ptr = self.lib.vip_map_buffer(buf)
            if not ptr:
                raise RuntimeError(f"vip_map_buffer returned NULL for {info['name']}")
            dtype = FORMAT_DTYPE.get(info["format"], np.float32)
            n = int(np.prod(info["shape"]))
            nbytes = n * np.dtype(dtype).itemsize
            raw = ctypes.string_at(ptr, nbytes)
            outs.append(np.frombuffer(raw, dtype=dtype, count=n).copy().reshape(info["shape"]))
            self.lib.vip_unmap_buffer(buf)
        return outs

    def _pack_input(self, rgb: np.ndarray) -> np.ndarray:
        info = self.inputs[0]
        shape = [s for s in info["shape"] if s > 0]
        h, w = rgb.shape[:2]
        if info["format"] == VIP_BUFFER_FORMAT_UINT8:
            chw = np.transpose(rgb, (2, 0, 1))
        else:
            chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        # VIPLite on A733 often reports C H W N  (3,640,640,1)
        if shape[:3] == [3, h, w] or (len(shape) >= 3 and shape[0] == 3):
            packed = chw.reshape(shape)
        elif shape[-3:] == [h, w, 3] or (len(shape) >= 3 and shape[-1] == 3):
            packed = rgb.reshape(shape) if info["format"] == VIP_BUFFER_FORMAT_UINT8 else (rgb.astype(np.float32) / 255.0).reshape(shape)
        else:
            packed = chw.reshape(-1)
        return np.ascontiguousarray(packed)

    def close(self) -> None:
        try:
            for buf in getattr(self, "in_bufs", []) + getattr(self, "out_bufs", []):
                self.lib.vip_destroy_buffer(buf)
            if getattr(self, "net", None):
                self.lib.vip_finish_network(self.net)
                self.lib.vip_destroy_network(self.net)
            self.lib.vip_destroy()
        except Exception:
            pass

    def __del__(self):
        self.close()


def decode_yolo26_6(
    outputs: list[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    nc: int = 80,
    strides: tuple[int, ...] = (8, 16, 32),
    layout: str = "chw",
    nms_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode 6 raw heads. layout='chw' is required on A733 VIPLite (Frigate/Radxa)."""
    flat = [_squeeze(o) for o in outputs]
    boxes_l, scores_l = _split_heads(flat, nc, imgsz, strides)
    all_boxes, all_scores = [], []
    for box, score, stride in zip(boxes_l, scores_l, strides):
        h = w = imgsz // stride
        box = _to_chw(box, 4, h, w, layout)
        score = _to_chw(score, nc, h, w, layout)
        box = box.reshape(4, -1)
        score = 1.0 / (1.0 + np.exp(-score.reshape(nc, -1)))
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchors = np.stack([(gx.reshape(-1) + 0.5), (gy.reshape(-1) + 0.5)], 0)
        lt, rb = box[:2], box[2:]
        xyxy = np.concatenate([anchors - lt, anchors + rb], 0) * stride
        all_boxes.append(xyxy)
        all_scores.append(score)
    boxes = np.concatenate(all_boxes, 1).T
    scores = np.concatenate(all_scores, 1).T
    cls = scores.argmax(1)
    best = scores.max(1)
    keep = best >= conf
    boxes, best, cls = boxes[keep], best[keep], cls[keep]
    if nms_fn is None or len(boxes) == 0:
        return boxes, best, cls
    kept = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for j in nms_fn(boxes[idx], best[idx], iou):
            kept.append(idx[j])
    kept = np.asarray(kept, dtype=int)
    return boxes[kept], best[kept], cls[kept]


def _squeeze(a: np.ndarray) -> np.ndarray:
    a = np.squeeze(a)
    return a.astype(np.float32, copy=False)


def _split_heads(outs: list[np.ndarray], nc: int, imgsz: int, strides: tuple[int, ...]):
    boxes, scores = [], []
    for stride in strides:
        hw = (imgsz // stride) ** 2
        box = score = None
        for o in outs:
            if o.size == 4 * hw and box is None:
                box = o
            elif o.size == nc * hw and score is None:
                score = o
        if box is None or score is None:
            raise RuntimeError(f"missing head for stride {stride}, output sizes={[o.size for o in outs]}")
        boxes.append(box)
        scores.append(score)
    return boxes, scores


def _to_chw(arr: np.ndarray, c: int, h: int, w: int, layout: str) -> np.ndarray:
    if arr.shape == (c, h, w):
        return arr
    if arr.shape == (h, w, c):
        return np.transpose(arr, (2, 0, 1))
    if arr.size != c * h * w:
        raise RuntimeError(f"bad head shape {arr.shape} expected {c}x{h}x{w}")
    if layout == "hwc":
        return np.transpose(arr.reshape(h, w, c), (2, 0, 1))
    return arr.reshape(c, h, w)


def probe() -> int:
    print(" /dev/vipcore:", Path("/dev/vipcore").exists())
    for name in ("libNBGlinker.so", "libVIPhal.so"):
        try:
            path = _find_lib(name)
            print(f" {name}: {path}")
        except FileNotFoundError as exc:
            print(f" {name}: MISSING ({exc})")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    raise SystemExit(probe())
