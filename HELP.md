# YOLO26 ONNX on Orange Pi Zero 3W

This folder can run YOLO26 in two ways:

| Path | Project | Model | Runtime |
|---|---|---|---|
| **ONNX CPU (this guide)** | [`Yolo26_ONNX`](Yolo26_ONNX) | `yolo26n_6.onnx` (6-head) or e2e `yolo26n.onnx` | ONNX Runtime CPU |
| NPU | [`Yolo26_NPU`](Yolo26_NPU) | ACUITY `*.nb` | VIPLite `/dev/vipcore` |

ONNX inference does **not** need VIPLite, `/dev/vipcore`, or ACUITY. Use `yolo26n` on 1–2 GB boards.

```
Already in this tree:  export/yolo26n_6.onnx   (6 heads: box_p3–p5 + cls_p3–p5)
Optional PC export:    prepare_onnx.py         → export_onnx/yolo26n.onnx  (1,300,6)
Board:  ./setup_onnx.sh  →  Yolo26_ONNX/build.sh  →  yolo26_onnx
```

The C++ / Python ONNX path now decodes **both** graphs. Your existing `export/yolo26*_6.onnx` files work; you do not need a second export.

NPU setup is in **[../HELP.md](../HELP.md)** Part 1–2. Do not pass this end-to-end ONNX to `convert_npu.sh`.

---

## Contents

1. [Prepare the ONNX environment (board)](#1-prepare-the-onnx-environment-board)
2. [Export ONNX on a PC](#2-export-onnx-on-a-pc)
3. [C++ project](#3-c-project)
4. [Run](#4-run)
5. [Problems](#5-problems)

---

# 1. Prepare the ONNX environment (board)

Do this **on the Orange Pi**, after you copy this folder (scp, USB, or git). Official Debian/Ubuntu aarch64 image.

## 1.1 Check the board

```bash
uname -m                          # must be aarch64
cat /proc/device-tree/model
free -h
```

ONNX Runtime CPU uses the Cortex-A cores + OpenBLAS. `/dev/vipcore` is **not** required.

On 1–2 GB boards, add swap before `apt` / `cmake` / `yolo26s`:

```bash
./setup_onnx.sh --swap
```

or:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 1.2 One-shot install

```bash
cd ~/yolo26-orangepi3w          # or wherever you copied this folder
chmod +x setup_onnx.sh run_onnx.sh Yolo26_ONNX/*.sh
./setup_onnx.sh                 # C++ tools + OpenCV + ONNX Runtime libs
# ./setup_onnx.sh --python      # also venv for infer.py
```

`setup_onnx.sh` does three things:

1. `apt` packages for the C++ app
2. Official ONNX Runtime **C/C++** tarball into `Yolo26_ONNX/third_party/onnxruntime`
3. Optional Python `onnxruntime` if you passed `--python`

## 1.3 C++ toolchain

If you prefer to install by hand:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake g++ pkg-config \
  git wget curl ca-certificates \
  libopencv-dev \
  libopencv-core-dev libopencv-imgproc-dev \
  libopencv-imgcodecs-dev libopencv-videoio-dev \
  libopencv-highgui-dev \
  libgl1 libglib2.0-0 libgtk-3-0 \
  libjpeg-dev libpng-dev libtiff-dev \
  libopenblas-dev libgomp1 libatomic1 \
  v4l-utils
```

| Package | Why |
|---|---|
| `build-essential`, `g++`, `cmake`, `pkg-config` | compile `yolo26_onnx` |
| `libopencv-*-dev` | image / camera / draw / `--save` |
| `libopenblas-dev`, `libgomp1` | ONNX Runtime CPU math |
| `libatomic1` | aarch64 C++ atomics |

```bash
g++ --version
cmake --version
pkg-config --modversion opencv4 || pkg-config --modversion opencv
```

You want CMake ≥ 3.10, C++17 g++, OpenCV 4.x.

Over SSH with no desktop, still install the OpenCV `-dev` packages. At run time use `--no-show --save result.jpg`.

## 1.4 ONNX Runtime C/C++ (required)

`apt` does **not** ship a usable `libonnxruntime.so` + C++ headers on this image. Download the official CPU package:

```bash
cd Yolo26_ONNX
chmod +x fetch_onnxruntime.sh
./fetch_onnxruntime.sh
```

That fetches **v1.24.4** (last official Linux **aarch64** tarball from Microsoft):

```
https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-aarch64-1.24.4.tgz
```

Result:

```
Yolo26_ONNX/third_party/onnxruntime/include/onnxruntime_cxx_api.h
Yolo26_ONNX/third_party/onnxruntime/lib/libonnxruntime.so
```

If the board has no network, download the `.tgz` on a PC, copy it over, then:

```bash
mkdir -p Yolo26_ONNX/third_party
tar -xzf onnxruntime-linux-aarch64-1.24.4.tgz -C Yolo26_ONNX/third_party
mv Yolo26_ONNX/third_party/onnxruntime-linux-aarch64-1.24.4 \
   Yolo26_ONNX/third_party/onnxruntime
```

Override the version or location:

```bash
export ONNXRUNTIME_VERSION=1.24.4
export ONNXRUNTIME_ROOT=/opt/onnxruntime
```

CMake also accepts `-DONNXRUNTIME_ROOT=...`.

## 1.5 Confirm the ONNX stack

```bash
uname -m
g++ --version | head -n1
cmake --version | head -n1
pkg-config --modversion opencv4
ls -l Yolo26_ONNX/third_party/onnxruntime/include/onnxruntime_cxx_api.h
ls -l Yolo26_ONNX/third_party/onnxruntime/lib/libonnxruntime.so*
```

Headers + `libonnxruntime.so` + OpenCV must be present before `./build.sh`.

## 1.6 Optional: Python (`infer.py`)

The C++ app does not need pip.

```bash
./setup_onnx.sh --python
source .venv/bin/activate
python infer.py --backend onnx --model export_onnx/yolo26n.onnx --source export_onnx/bus.jpg --no-save
```

---

# 2. Export ONNX on a PC

Do **not** export on a 1 GB board. Use a PC (Windows or Linux).

This is the **end-to-end detect** graph (boxes + score + class). It is **not** the 6-head NPU ONNX from `export_npu.py`.

**Windows**

```bat
python -m pip install -U -r requirements-prepare.txt
python prepare_onnx.py --model yolo26n.pt --imgsz 640 --outdir export_onnx
```

Or `prepare_onnx.bat`.

**Linux**

```bash
python3 -m pip install -U -r requirements-prepare.txt
python3 prepare_onnx.py --model yolo26n.pt --imgsz 640 --outdir export_onnx
```

| File | Role |
|---|---|
| `export_onnx/yolo26n.onnx` | standard YOLO26 ONNX (opset 12, NMS on) |
| `export_onnx/bus.jpg` | sample image |
| `export_onnx/MANIFEST.txt` | copy/run reminder |

`--imgsz` must match later `--imgsz` (default **640**). Prefer `yolo26n.pt` on the 3W. `yolo26s.pt` needs more RAM and is slower on CPU.

Copy `export_onnx/` to the board (next to `Yolo26_ONNX/`).

---

# 3. C++ project

Folder: [`Yolo26_ONNX`](Yolo26_ONNX)

CMake target: `yolo26_onnx` (C++17, OpenCV, ONNX Runtime CPU).

| Source | Role |
|---|---|
| `src/main.cpp` | image / video / camera runner |
| `src/onnx_engine.cpp` | ONNX Runtime session + NCHW/NHWC input |
| `src/yolo26_post.cpp` | letterbox, e2e / raw-head decode, NMS, draw |

Finish [Part 1](#1-prepare-the-onnx-environment-board) before building.

```bash
cd Yolo26_ONNX
chmod +x build.sh fetch_onnxruntime.sh
./build.sh
```

Or:

```bash
cd Yolo26_ONNX
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$PWD/../third_party/onnxruntime
cmake --build . -j$(nproc)
```

On a 1 GB board use `cmake --build . -j1`.

Binary: `Yolo26_ONNX/build/yolo26_onnx`

---

# 4. Run

```bash
cd ~/yolo26-orangepi3w

# picks export/yolo26n_6.onnx (or export_onnx/yolo26n.onnx) + a calib image
./run_onnx.sh
./run_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg

# explicit files + extra flags
./run_onnx.sh export_onnx/yolo26n.onnx export_onnx/bus.jpg --threads 2 --conf 0.3
```

Or the binary:

```bash
cd Yolo26_ONNX/build
export LD_LIBRARY_PATH=../third_party/onnxruntime/lib:$LD_LIBRARY_PATH

# image, headless
./yolo26_onnx /path/to/yolo26n.onnx /path/to/bus.jpg --no-show --save result.jpg

# image, show window (press q / ESC)
./yolo26_onnx /path/to/yolo26n.onnx /path/to/bus.jpg

# USB camera
./yolo26_onnx /path/to/yolo26n.onnx 0 --threads 2

# video
./yolo26_onnx /path/to/yolo26n.onnx clip.mp4 --save out.mp4 --no-show

# FPS on one image
./yolo26_onnx /path/to/yolo26n.onnx bus.jpg --loop 50 --no-show
```

| Flag | Default | Meaning |
|---|---|---|
| `--imgsz N` | 640 | Must match the exported ONNX |
| `--conf F` | 0.25 | Score threshold |
| `--nms F` | 0.45 | Class-aware IoU NMS (raw-head graphs) |
| `--nc N` | 80 | COCO class count |
| `--threads N` | 4 | ONNX Runtime intra-op threads (use 2 on 1 GB) |
| `--loop N` | 1 | Repeat one image for timing |
| `--no-show` | off | Headless (SSH / no display) |
| `--save PATH` | `result.jpg` | Annotated image or video |

Over SSH without a desktop, always pass `--no-show`.

Expected CPU time on a 3W for **yolo26n @ 640**: about **0.5–2 s / frame** depending on RAM and `--threads`. This is CPU, not NPU.

---

# 5. Problems

| Symptom | What to do |
|---|---|
| `cmake` / OpenCV not found | [1.3](#13-c-toolchain) or `./setup_onnx.sh` |
| `ONNX Runtime not found` | [1.4](#14-onnx-runtime-cc-required) `./Yolo26_ONNX/fetch_onnxruntime.sh` |
| `libonnxruntime.so: cannot open` | `export LD_LIBRARY_PATH=Yolo26_ONNX/third_party/onnxruntime/lib:$LD_LIBRARY_PATH` |
| `wget` / GitHub fails | Download the aarch64 `.tgz` on a PC and extract as in 1.4 |
| Board OOM while compiling | `./setup_onnx.sh --swap`, then `cmake --build . -j1` |
| Board OOM while running | `--threads 2`, use `yolo26n`, not `s/m/l/x` |
| No detections / huge boxes | `--imgsz` must match export (default 640). Use `prepare_onnx.py`, not `export_npu.py` |
| ACUITY / NPU errors | Wrong path. This ONNX is for CPU only. NPU needs [../HELP.md](../HELP.md#2-yolo26-c-project) |
| OpenCV window fails over SSH | `--no-show --save result.jpg` |
| Slow | Normal on CPU. Use `yolo26n`, `--imgsz 320` (re-export), or the NPU `.nb` app |

## Checklist

1. PC: `python prepare_onnx.py --model yolo26n.pt --outdir export_onnx`
2. Copy this folder + `export_onnx/` to the Orange Pi
3. Board: `./setup_onnx.sh` (add `--swap` on 1 GB)
4. `cd Yolo26_ONNX && ./build.sh`
5. `./run_onnx.sh` or `./build/yolo26_onnx model.onnx bus.jpg --no-show --save result.jpg`
