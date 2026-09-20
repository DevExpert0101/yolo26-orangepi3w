#!/usr/bin/env python3
"""YOLO26 inference for Orange Pi Zero 3W.

Backends:
  --backend auto   pick from the model path
  --backend npu    VIPLite on /dev/vipcore (*.nb from convert_npu.sh)
  --backend onnx   onnxruntime CPU
  --backend ncnn   Ultralytics NCNN CPU

Sources:
  image / folder / video file / camera index (0) / CSI via --csi
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 inference on Orange Pi 3W")
    parser.add_argument("--model", required=True, help="ONNX file or NCNN model directory")
    parser.add_argument("--source", default="0", help="Image, folder, video, or camera index")
    parser.add_argument("--backend", choices=("auto", "npu", "onnx", "ncnn"), default="auto")
    parser.add_argument("--layout", choices=("chw", "hwc"), default="chw", help="NPU output layout")
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="cpu or vulkan:0 for NCNN")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save", default="runs/detect", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Open a preview window")
    parser.add_argument("--csi", action="store_true", help="Use picamera2 / CSI camera")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def detect_backend(model: str, requested: str) -> str:
    if requested != "auto":
        return requested
    path = Path(model)
    if path.suffix.lower() == ".nb":
        return "npu"
    if path.is_dir() or path.name.endswith("_ncnn_model") or (path / "model.param").exists():
        return "ncnn"
    return "onnx"


def letterbox(image: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


def scale_boxes(boxes: np.ndarray, ratio: float, pad: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    h, w = shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def parse_onnx_output(raw, conf: float, iou: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = raw[0] if isinstance(raw, (list, tuple)) else raw
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected ONNX output shape: {getattr(raw, 'shape', type(raw))}")

    # YOLO26 end-to-end: (300, 6) = xyxy, score, class
    if pred.shape[-1] == 6:
        boxes, scores, classes = pred[:, :4], pred[:, 4], pred[:, 5].astype(int)
        keep = scores >= conf
        return boxes[keep], scores[keep], classes[keep]

    # Traditional YOLO: (4+nc, N) or (N, 4+nc)
    if pred.shape[0] in (4 + 80, 5 + 80) or pred.shape[0] < pred.shape[1]:
        pred = pred.T
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    classes = class_scores.argmax(1)
    scores = class_scores.max(1)
    keep = scores >= conf
    boxes_xywh, scores, classes = boxes_xywh[keep], scores[keep], classes[keep]
    boxes = np.stack(
        [
            boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
            boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
            boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2,
            boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2,
        ],
        axis=1,
    )
    kept = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        for j in nms(boxes[idx], scores[idx], iou):
            kept.append(idx[j])
    kept = np.array(kept, dtype=int) if kept else np.array([], dtype=int)
    return boxes[kept], scores[kept], classes[kept]


def draw(image: np.ndarray, boxes, scores, classes) -> np.ndarray:
    out = image.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        name = COCO80[int(cls)] if 0 <= int(cls) < len(COCO80) else str(int(cls))
        label = f"{name} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


class OnnxEngine:
    def __init__(self, model: str, imgsz: int, conf: float, iou: float, threads: int):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def __call__(self, bgr: np.ndarray):
        lb, ratio, pad = letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None]
        raw = self.session.run(None, {self.input_name: tensor})
        boxes, scores, classes = parse_onnx_output(raw, self.conf, self.iou)
        boxes = scale_boxes(boxes, ratio, pad, bgr.shape[:2]) if len(boxes) else boxes
        return boxes, scores, classes


class NpuEngine:
    def __init__(self, model: str, imgsz: int, conf: float, iou: float, layout: str, nc: int):
        from npu_runtime import VipLite, decode_yolo26_6

        self.vip = VipLite(model)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.layout = layout
        self.nc = nc
        self.decode = decode_yolo26_6

    def __call__(self, bgr: np.ndarray):
        lb, ratio, pad = letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        raw = self.vip.infer(rgb)
        boxes, scores, classes = self.decode(
            raw, self.imgsz, self.conf, self.iou, nc=self.nc, layout=self.layout, nms_fn=nms
        )
        boxes = scale_boxes(boxes, ratio, pad, bgr.shape[:2]) if len(boxes) else boxes
        return boxes, scores, classes


class UltralyticsEngine:
    def __init__(self, model: str, imgsz: int, conf: float, iou: float, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "NCNN backend needs ultralytics. On the board run:  ./setup_board.sh --full"
            ) from exc
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

    def __call__(self, bgr: np.ndarray):
        result = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return boxes, scores, classes


def iter_sources(source: str, csi: bool):
    if csi:
        from picamera2 import Picamera2

        cam = Picamera2()
        cam.preview_configuration.main.size = (1280, 720)
        cam.preview_configuration.main.format = "RGB888"
        cam.configure("preview")
        cam.start()
        try:
            while True:
                frame = cam.capture_array()
                yield "csi", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        finally:
            cam.close()
        return

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"cam{source}", frame
        finally:
            cap.release()
        return

    path = Path(source)
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for file in sorted(p for p in path.iterdir() if p.suffix.lower() in exts):
            image = cv2.imread(str(file))
            if image is None:
                print(f"Skip unreadable {file}")
                continue
            yield file.name, image
        return

    if path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        cap = cv2.VideoCapture(str(path))
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"{path.stem}_{idx:06d}", frame
                idx += 1
        finally:
            cap.release()
        return

    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Cannot read {path}")
    yield path.name, image


def main() -> int:
    args = parse_args()
    backend = detect_backend(args.model, args.backend)
    print(f"backend={backend}  model={args.model}  source={args.source}  imgsz={args.imgsz}")

    if backend == "npu":
        engine = NpuEngine(args.model, args.imgsz, args.conf, args.iou, args.layout, args.nc)
    elif backend == "onnx":
        engine = OnnxEngine(args.model, args.imgsz, args.conf, args.iou, args.threads)
    else:
        engine = UltralyticsEngine(args.model, args.imgsz, args.conf, args.iou, args.device)

    save_dir = Path(args.save)
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)

    times: list[float] = []
    count = 0
    stream = args.csi or args.source.isdigit()
    for name, frame in iter_sources(args.source, args.csi):
        t0 = time.perf_counter()
        boxes, scores, classes = engine(frame)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        count += 1
        annotated = draw(frame, boxes, scores, classes)
        fps = 1000.0 / (sum(times[-30:]) / min(len(times), 30))
        cv2.putText(
            annotated,
            f"{ms:.1f} ms  {fps:.1f} FPS  {len(boxes)} det",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        print(f"{name}: {len(boxes)} det, {ms:.1f} ms")
        if args.show:
            cv2.imshow("YOLO26 OrangePi 3W", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if not args.no_save and not stream:
            cv2.imwrite(str(save_dir / f"{Path(name).stem}.jpg"), annotated)
        elif not args.no_save and stream and count == 1:
            cv2.imwrite(str(save_dir / "camera_preview.jpg"), annotated)

    if args.show:
        cv2.destroyAllWindows()
    if times:
        print(
            f"done: {count} frames, "
            f"avg {sum(times)/len(times):.1f} ms, "
            f"min {min(times):.1f}, max {max(times):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
