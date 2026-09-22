# YOLO26 on Orange Pi Zero 3W

Board: Allwinner **A733**, Vivante **VIP9000** NPU (3 TOPS). This folder has two C++ apps.

| Path | Folder | Model | Runtime |
|---|---|---|---|
| **ONNX CPU** | [`Yolo26_ONNX`](Yolo26_ONNX) | `*.onnx` | ONNX Runtime CPU |
| **NPU** | [`Yolo26_NPU`](Yolo26_NPU) | ACUITY `*.nb` | VIPLite `/dev/vipcore` |

```
PC:   yolo26n.pt ──export_npu.py──► yolo26n_6.onnx ──ACUITY──► yolo26n_fp16.nb
                                         │
                                         ├── Yolo26_ONNX (CPU, no VIPLite)
                                         └── Yolo26_NPU  (needs the .nb)
```

| File | Use it with | Do not |
|---|---|---|
| `export/yolo26n_6.onnx` | `yolo26_onnx`, `infer.py --backend onnx`, `convert_npu.sh` | `yolo26_npu` |
| `export_onnx/yolo26n.onnx` (e2e 300×6) | `yolo26_onnx` only | `convert_npu.sh` / `yolo26_npu` |
| `export_nb/yolo26n_fp16.nb` | `yolo26_npu`, `infer.py --backend npu` | `yolo26_onnx` |

`.pt` / `.onnx` / `.nb` are gitignored. `git pull` on the board does **not** fetch weights. Copy `export/` and `export_nb/` with scp/USB.

Prefer **yolo26n** on 1–2 GB boards. Start NPU with `yolo26n_fp16.nb`.

Repo: https://github.com/DevExpert0101/yolo26-orangepi3w

---

## Contents

1. [Orange Pi environment](#1-orange-pi-environment)
2. [ONNX C++ project (CPU)](#2-onnx-c-project-cpu)
3. [Convert models to NPU](#3-convert-models-to-npu)
4. [NPU C++ project](#4-npu-c-project)
5. [Measure inference time](#5-measure-inference-time)
6. [Problems](#6-problems)

---

# 1. Orange Pi environment

Do this **on the board** after you copy this folder (`~/Documents/yolo26-orangepi3w` in the examples). Use the **official Orange Pi Debian/Ubuntu** image. Mainline / Armbian often has no NPU userspace — `apt` cannot add `/dev/vipcore` or VIPLite.

## 1.1 Check the board

```bash
uname -m                          # must be aarch64
cat /proc/device-tree/model
free -h
ls -l /dev/vipcore                # required for NPU; not required for ONNX CPU
```

If `/dev/vipcore` is missing, flash the vendor image before any NPU work.

## 1.2 Swap (1–2 GB boards)

Needed before `apt`, `cmake`, or `yolo26s`:

```bash
cd ~/Documents/yolo26-orangepi3w
./setup_board.sh --swap
# or: ./setup_onnx.sh --swap
```

Manual:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 1.3 C++ toolchain + OpenCV (both apps)

Both `yolo26_onnx` and `yolo26_npu` need g++, CMake, OpenCV.

**One-shot (NPU + OpenCV + C++):**

```bash
cd ~/Documents/yolo26-orangepi3w
chmod +x setup_board.sh setup_onnx.sh
./setup_board.sh                 # apt + OpenCV + VIPLite probe
# ./setup_board.sh --swap
# ./setup_board.sh --python      # venv for infer.py
```

**ONNX only** (no VIPLite):

```bash
./setup_onnx.sh                  # apt + OpenCV + official ORT aarch64 libs
# ./setup_onnx.sh --python
```

Manual apt (same packages both apps use):

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
| `build-essential`, `g++`, `cmake`, `pkg-config` | compile both C++ apps |
| `libopencv-*-dev` | image / camera / draw / `--save` |
| `libopenblas-dev`, `libgomp1` | ORT CPU math |
| `libatomic1` | aarch64 C++ atomics |

```bash
g++ --version                     # C++17
cmake --version                   # ≥ 3.10
pkg-config --modversion opencv4 || pkg-config --modversion opencv
```

Over SSH with no desktop, still install the OpenCV `-dev` packages. At run time use `--no-show --save result.jpg`.

## 1.4 NPU device permission

Skip this section if you only run ONNX CPU.

```bash
sudo chmod 666 /dev/vipcore
echo 'KERNEL=="vipcore", MODE="0666"' | sudo tee /etc/udev/rules.d/99-vipcore.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 1.5 VIPLite libraries (NPU only)

`yolo26_npu` links **`libNBGlinker.so`** and **`libVIPhal.so`**. They come with the Orange Pi / Allwinner NPU userspace, not `apt`. `./setup_board.sh` copies whatever it finds into `Yolo26_NPU/lib/`.

```bash
sudo find /usr /opt /lib /home -name 'libNBGlinker.so' -o -name 'libVIPhal.so' 2>/dev/null
ldconfig -p | grep -E 'NBGlinker|VIPhal'
```

Typical locations: `/usr/lib/`, `/usr/lib/aarch64-linux-gnu/`, `/opt/yolov5/`, or the SDK:

```
linux_aw_npu/.../common/npuruntime/lib_linux_aarch64/A733/
```

If they are not on the loader path:

```bash
mkdir -p ~/Documents/yolo26-orangepi3w/Yolo26_NPU/lib
cp /opt/yolov5/libNBGlinker.so /opt/yolov5/libVIPhal.so \
   ~/Documents/yolo26-orangepi3w/Yolo26_NPU/lib/
# copy any other libNBG* / libVIP* sitting next to those two
```

Or system-wide:

```bash
sudo cp libNBGlinker.so libVIPhal.so /usr/local/lib/
sudo ldconfig
```

This project already bundles `Yolo26_NPU/include/vip_lite.h`. You do not need the SDK header unless you set `-DVIPLITE_ROOT=...`.

## 1.6 Confirm

**ONNX CPU**

```bash
ls Yolo26_ONNX/third_party/onnxruntime/include/onnxruntime_cxx_api.h
ls Yolo26_ONNX/third_party/onnxruntime/lib/libonnxruntime.so
```

**NPU**

```bash
ls -l /dev/vipcore
ls Yolo26_NPU/lib/libNBGlinker.so Yolo26_NPU/lib/libVIPhal.so
# or: ldconfig -p | grep -E 'NBGlinker|VIPhal'
```

## 1.7 Optional: Python / camera

C++ apps do not need pip.

```bash
./setup_board.sh --python        # infer.py + npu_runtime.py --probe
# ./setup_board.sh --full        # also ultralytics + ncnn (heavy)

v4l2-ctl --list-devices          # USB camera → source 0
```

The vendor `/opt/yolov5/yolov5` binary may want `libopencv_*so.4.5` while apt provides `*.4.5d`. That demo is YOLOv5. It is **not** required for these apps.

## 1.8 Shared CMake (NPU)

| Option | When |
|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | default |
| `-DVIPLITE_ROOT=/opt/npu` | headers/libs not in `/usr` |
| `-DVIP_INIT_HAS_SIZE=ON` | `vip_init` link/runtime mismatch |

```bash
export VIPLITE_ROOT=/opt/npu
export VIP_INIT_HAS_SIZE=ON
cd Yolo26_NPU && ./build.sh
export LD_LIBRARY_PATH=/usr/lib:Yolo26_NPU/lib:$LD_LIBRARY_PATH
```

---

# 2. ONNX C++ project (CPU)

Folder: [`Yolo26_ONNX`](Yolo26_ONNX). No VIPLite, no `/dev/vipcore`, no ACUITY.

The decoder accepts:

- **6-head** `export/yolo26n_6.onnx` — `box_p3/p4/p5` + `cls_p3/p4/p5` (already in this tree)
- **e2e** `(1,300,6)` from `prepare_onnx.py`
- **raw** `(1,84,8400)` one-to-many

CMake target: `yolo26_onnx` (C++17, OpenCV, ONNX Runtime CPU).

| Source | Role |
|---|---|
| `src/main.cpp` | image / video / camera + timing |
| `src/onnx_engine.cpp` | ORT session, all outputs |
| `src/yolo26_post.cpp` | letterbox, 6-head / e2e / raw decode, NMS |

## 2.1 ONNX Runtime C/C++ on the board

`apt` does not ship usable `libonnxruntime.so` + C++ headers. `./setup_onnx.sh` downloads official **v1.24.4** aarch64:

```
https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-aarch64-1.24.4.tgz
```

into `Yolo26_ONNX/third_party/onnxruntime/`.

Offline: copy the `.tgz` from a PC, then:

```bash
mkdir -p Yolo26_ONNX/third_party
tar -xzf onnxruntime-linux-aarch64-1.24.4.tgz -C Yolo26_ONNX/third_party
mv Yolo26_ONNX/third_party/onnxruntime-linux-aarch64-1.24.4 \
   Yolo26_ONNX/third_party/onnxruntime
```

```bash
export ONNXRUNTIME_VERSION=1.24.4
export ONNXRUNTIME_ROOT=/opt/onnxruntime   # or cmake -DONNXRUNTIME_ROOT=...
```

A log line `GPU device discovery failed: /sys/class/drm/...` is normal on this board. Ignore it.

## 2.2 Optional: export e2e ONNX on a PC

Do **not** export on a 1 GB board. The 6-head file `export/yolo26n_6.onnx` is enough for CPU; this step is only if you want the `(1,300,6)` graph.

**Windows**

```bat
python -m pip install -U -r requirements-prepare.txt
python prepare_onnx.py --model yolo26n.pt --imgsz 640 --outdir export_onnx
```

Or `prepare_onnx.bat`. `--raw` exports one-to-many `(1,84,8400)` instead of e2e.

**Linux**

```bash
python3 -m pip install -U -r requirements-prepare.txt
python3 prepare_onnx.py --model yolo26n.pt --imgsz 640 --outdir export_onnx
```

| File | Role |
|---|---|
| `export_onnx/yolo26n.onnx` | YOLO26 e2e (`nms=False`, opset 12) |
| `export_onnx/bus.jpg` | sample image |

Copy `export_onnx/` to the board. Do **not** pass this file to `convert_npu.sh`.

## 2.3 Build

```bash
cd ~/Documents/yolo26-orangepi3w
./setup_onnx.sh
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
cmake --build . -j$(nproc)          # 1 GB: -j1
```

Binary: `Yolo26_ONNX/build/yolo26_onnx`

## 2.4 Run

```bash
cd ~/Documents/yolo26-orangepi3w
chmod +x run_onnx.sh bench_onnx.sh

./run_onnx.sh
./run_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg
./run_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg --threads 2 --conf 0.3
```

Or the binary:

```bash
cd Yolo26_ONNX/build
export LD_LIBRARY_PATH=../third_party/onnxruntime/lib:$LD_LIBRARY_PATH

./yolo26_onnx ../../export/yolo26n_6.onnx ../../export/calib/bus.jpg --no-show --save result.jpg
./yolo26_onnx ../../export/yolo26n_6.onnx ../../export/calib/bus.jpg
./yolo26_onnx ../../export/yolo26n_6.onnx 0 --threads 2
./yolo26_onnx ../../export/yolo26n_6.onnx clip.mp4 --save out.mp4 --no-show
```

| Flag | Default | Meaning |
|---|---|---|
| `--imgsz N` | 640 | Must match export |
| `--conf F` | 0.25 | Score threshold |
| `--nms F` | 0.45 | Class-aware IoU NMS (6-head / raw) |
| `--nc N` | 80 | COCO classes |
| `--threads N` | 4 | ORT intra-op (use 2 on 1 GB) |
| `--loop N` | 1 | Timed repeats after warmup |
| `--warmup N` | 3 if `--loop>1` | Discarded runs |
| `--no-show` | off | Required over SSH |
| `--save PATH` | `result.jpg` | Annotated image or video |

Python:

```bash
./setup_onnx.sh --python
source .venv/bin/activate
python infer.py --backend onnx --model export/yolo26n_6.onnx --source export/calib/bus.jpg
```

Expected CPU time, yolo26n @ 640: about **0.5–2 s / frame**. Use the NPU app for real FPS.

---

# 3. Convert models to NPU

The NPU does **not** run `.pt` or a default YOLO26 ONNX. ACUITY `pegasus` compiles a **6-head** ONNX to an A733 `.nb`. This step is on a **PC / WSL**, never on the 3W.

```
yolo26n.pt  ──export_npu.py──►  yolo26n_6.onnx  ──convert_npu.sh──►  yolo26n_fp16.nb
```

Default compile is **FP16** (`--quant fp16`). ACUITY `--dtype float` becomes an FP16 NBG on VIP9000 (no FP32 MACs). INT8 is still available with `--quant pcq`. Target:

```
VIP9000NANODI_PLUS_PID0X1000003B
```

Convert follows the Allwinner yolo11 zoo and the working A733 path in [Frigate #23418](https://github.com/blakeblackshear/frigate/discussions/23418) / [CONVERSION.md](https://github.com/unnamedwild-ux/frigate_npu_vivante/blob/master/CONVERSION.md): `pegasus import onnx` → `generate inputmeta` + `generate postprocess-file` → `IMAGE_RGB` preproc (HWC uint8 0–255) and float32 outputs. Decode the 6 heads as **CHW** even though `vip_query_output` prints HWC. Older scripts left `add_preproc_node: false` and skipped postprocess, which produced garbage boxes. **Reconvert** after this fix; delete leftover `export/*.json` (except `*_npu.json`), `export/*.data`, `export/*.quantize`, and `export/wksp/` if convert is not doing a clean import.

| On the PC | Role |
|---|---|
| `export/yolo26n_6.onnx` | 6-head graph (also runs on ONNX CPU) |
| `export/yolo26n_6_inputmeta.yml` | ACUITY input meta (`IMAGE_RGB`, `add_preproc_node: true`, scale 1/255) |
| `export/yolo26n_6_postprocess_file.yml` | ACUITY postproc (`add_postproc_node: true` → float32 outputs) |
| `export/yolo26n_6_npu.json` | imgsz / class count sidecar from `export_npu.py` (not the ACUITY graph) |
| `export/dataset.txt` + `export/calib/` | quantization images |
| `export_nb/yolo26n_fp16.nb` | FP16 NBG (preferred) |
| `export_nb/yolo26n.nb` | INT8 PCQ NBG |
| `export/yolo26n_6_fp16_a733.nb` | same FP16 graph from convert |

## 3.1 Export 6-head ONNX (PC)

YOLO26’s end-to-end NMS head is not NPU-legal. `export_npu.py` strips it and writes:

```
box_p3 (1,4,80,80)  box_p4 (1,4,40,40)  box_p5 (1,4,20,20)
cls_p3 (1,80,80,80) cls_p4 (1,80,40,40) cls_p5 (1,80,20,20)
```

**Windows**

```bat
python -m pip install -U ultralytics onnx onnxslim
python export_npu.py --model yolo26n.pt --imgsz 640 --outdir export
```

Or `prepare.bat`. `--all` exports n/s/m/l/x.

**Linux**

```bash
python3 -m pip install -U ultralytics onnx onnxslim
python3 export_npu.py --model yolo26n.pt --imgsz 640 --outdir export
```

`--imgsz` must match later `--imgsz` (default **640**).

Do not use `prepare_onnx.py` as ACUITY input.

## 3.2 ACUITY toolkit (Linux or WSL, x86_64)

Not native Windows, not the 3W. From Orange Pi / KickPi `linux_aw_npu` (or VeriSilicon ACUITY):

- `acuity-toolkit-whl-6.30.22` (or `acuity-toolkit-binary-*`) so `bin/pegasus` exists
- `VivanteIDE5.11.0/cmdtools` (required for `pegasus export ovxlib`)

The wheel toolkit often wants Python 3.8. The binary toolkit is easier if pip/TF fights you.

```bash
export ACUITY_PATH=$HOME/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=$HOME/Vivante_IDE/VivanteIDE5.11.0/cmdtools
export PATH=$ACUITY_PATH:$PATH
pegasus help
```

Do **not** run `convert_npu.sh` with sudo (ACUITY lives in your home; sudo looks in `/root`).

## 3.3 Compile the `.nb`

From this folder on Linux/WSL:

```bash
chmod +x convert_npu.sh run_pegasus.sh
export ACUITY_PATH=$HOME/acuity-toolkit-whl-6.30.22/bin
export VIV_SDK=$HOME/Vivante_IDE/VivanteIDE5.11.0/cmdtools
export PATH=$ACUITY_PATH:$PATH

./convert_npu.sh --onnx export/yolo26n_6.onnx                 # FP16 (default)
# ./convert_npu.sh --onnx export/yolo26n_6.onnx --quant pcq    # INT8
# ./convert_npu.sh --onnx export/yolo26n_6.onnx --native
# ./convert_npu.sh --onnx export/yolo26n_6.onnx --zoo /path/to/awnpu_model_zoo
```

| Flag | Meaning |
|---|---|
| `--onnx PATH` | 6-head ONNX from `export_npu.py` |
| `--name NAME` | stem (default: ONNX filename) |
| `--imgsz N` | default from `NAME_npu.json` or 640 |
| `--quant TYPE` | `fp16` (default) or `pcq` |
| `--native` | host `pegasus` (implied if on `PATH`) |
| `--docker` | `ubuntu-npu:v2.0.10.2` (optional) |
| `--sdk DIR` | `linux_aw_npu` tree |
| `--zoo DIR` | optional `awnpu_model_zoo` wrappers |

```bash
export NPU_DOCKER_IMAGE=ubuntu-npu:v2.0.10.2
./convert_npu.sh --docker --onnx export/yolo26n_6.onnx
```

Success looks like `export/yolo26n_6_fp16_a733.nb` and `export_nb/yolo26n_fp16.nb`. INT8 names use `pcq` instead of `fp16`.

After convert, `export/yolo26n_6_inputmeta.yml` must contain `add_preproc_node: true` and `preproc_type: IMAGE_RGB`. If it still says `false`, convert did not patch meta — do not use that `.nb`.

The NPU app writes **HWC uint8 0–255** into the NBG (same as `yolo11_6_pre.cpp`). Rebuild `yolo26_npu` on the board after `git pull`.

FP16 skips INT8 calibration and usually matches ONNX boxes much closer than PCQ. It is larger and a bit slower than PCQ (VIP9000 peak TOPS is INT8).

## 3.4 Copy to the board

`.nb` is not in git.

```powershell
scp D:\1\work\orangePi\yolo26-orangepi3w\export_nb\yolo26n_fp16.nb `
  orangepi@<board-ip>:~/Documents/yolo26-orangepi3w/export_nb/
```

Also copy a test image if needed: `export/calib/bus.jpg`.

```bash
ls -lh ~/Documents/yolo26-orangepi3w/export_nb/yolo26n_fp16.nb
```

---

# 4. NPU C++ project

Folder: [`Yolo26_NPU`](Yolo26_NPU). VIPLite runs the `.nb`. CPU does letterbox, dist2bbox, and NMS.

Expected NPU time for YOLO26s @ 640 (Radxa A733 reference): about **35 ms** infer, ~28 FPS end-to-end. yolo26n is lighter.

CMake target: `yolo26_npu` (C++17, OpenCV, `NBGlinker`, `VIPhal`).

| Source | Role |
|---|---|
| `src/main.cpp` | image / video / camera + timing |
| `src/viplite_engine.cpp` | VIPLite load / infer |
| `src/yolo26_post.cpp` | letterbox, dist2bbox, NMS, draw |
| `include/vip_lite.h` | bundled VIPLite API |

Finish [Part 1](#1-orange-pi-environment) (especially 1.4–1.5) before building.

## 4.1 Build

```bash
cd ~/Documents/yolo26-orangepi3w
./setup_board.sh
cd Yolo26_NPU
chmod +x build.sh
./build.sh
```

Or:

```bash
cd Yolo26_NPU
mkdir -p lib build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

If `vip_init` fails to link or run:

```bash
cd Yolo26_NPU/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DVIP_INIT_HAS_SIZE=ON
cmake --build . -j$(nproc)
```

Binary: `Yolo26_NPU/build/yolo26_npu`

## 4.2 Run

```bash
cd ~/Documents/yolo26-orangepi3w
chmod +x run_npu.sh bench_npu.sh

./run_npu.sh
./run_npu.sh export_nb/yolo26n_fp16.nb export/calib/bus.jpg
./run_npu.sh export_nb/wksp/yolo26n_6_fp16_nbg_unify/network_binary.nb export/calib/bus.jpg
```

Or the binary (SSH: always `--no-show`):

```bash
cd Yolo26_NPU/build
export LD_LIBRARY_PATH=../lib:/usr/lib:$LD_LIBRARY_PATH

./yolo26_npu ../../export_nb/yolo26n_fp16.nb ../../export/calib/bus.jpg --no-show --save ../../result_npu.jpg
./yolo26_npu ../../export_nb/yolo26n_fp16.nb ../../export/calib/bus.jpg
./yolo26_npu ../../export_nb/yolo26n_fp16.nb 0
./yolo26_npu ../../export_nb/yolo26n_fp16.nb clip.mp4 --save out.mp4 --no-show
```

| Flag | Default | Meaning |
|---|---|---|
| `--imgsz N` | 640 | Must match the exported ONNX |
| `--conf F` | 0.25 | Score threshold |
| `--nms F` | 0.45 | Class-aware IoU NMS |
| `--nc N` | 80 | COCO classes |
| `--loop N` | 1 | Timed repeats after warmup |
| `--warmup N` | 3 if `--loop>1` | Discarded runs |
| `--no-show` | off | Required over SSH |
| `--save PATH` | `result.jpg` | Annotated image or video |

Python:

```bash
./setup_board.sh --python
source .venv/bin/activate
python infer.py --backend npu --model export_nb/yolo26n_fp16.nb --source export/calib/bus.jpg
```

PCQ `.nb` files are INT8. The app writes `pixel-128` on INT8 inputs and dequants outputs with `(q - zp) * scale`.

VIPLite **reports** output sizes as `[H,W,C,1]` but the buffer is **CHW**. Reading them as HWC labels a car as person with a full-frame box ([Frigate discussion #23418](https://github.com/blakeblackshear/frigate/discussions/23418)). Decode defaults to CHW. `--layout hwc` is only a debug override.

Do not pass `yolo26n_6.onnx` to `yolo26_npu`.

---

# 5. Measure inference time

A single `./run_*.sh` is not a reliable time (first call includes runtime load). Use the bench scripts: **5 warmup + 30 timed runs**.

The number to quote:

| Script | Line | What it is |
|---|---|---|
| `./bench_onnx.sh` | **ORT infer** | ONNX Runtime `session.Run` |
| `./bench_npu.sh` | **NPU infer** | VIPLite `vip_run` |

End-to-end also includes letterbox and CPU NMS.

```bash
cd ~/Documents/yolo26-orangepi3w
git pull
chmod +x bench_onnx.sh bench_npu.sh run_onnx.sh run_npu.sh

# rebuild after pull
cd Yolo26_ONNX && ./build.sh && cd ..
cd Yolo26_NPU  && ./build.sh && cd ..

./bench_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg --loop 30
./bench_npu.sh  export_nb/yolo26n_fp16.nb export/calib/bus.jpg --loop 30
```

Example NPU summary:

```
--- NPU timing (30 timed runs, 5 warmup discarded) ---
NPU infer   avg ...  min ...  max ... ms   (... FPS)
preprocess  avg ... ms
postprocess avg ... ms
end-to-end  avg ...  min ...  max ... ms   (... FPS)
```

```bash
./bench_onnx.sh --threads 2 --loop 50
./bench_npu.sh --loop 50
```

---

# 6. Problems

| Symptom | What to do |
|---|---|
| `/dev/vipcore` missing | Official Orange Pi image ([1.1](#11-check-the-board)) |
| `permission denied` on `/dev/vipcore` | [1.4](#14-npu-device-permission) |
| `cmake` / `g++` / OpenCV not found | [1.3](#13-c-toolchain--opencv-both-apps) |
| `ONNX Runtime not found` | `./setup_onnx.sh` or `Yolo26_ONNX/fetch_onnxruntime.sh` |
| `libonnxruntime.so: cannot open` | `export LD_LIBRARY_PATH=Yolo26_ONNX/third_party/onnxruntime/lib:$LD_LIBRARY_PATH` |
| input rank 0 / load fail (ORT) | Rebuild after `git pull` (TypeInfo lifetime fix) |
| `libNBGlinker.so: cannot open` | Copy `.so` into `Yolo26_NPU/lib` ([1.5](#15-viplite-libraries-npu-only)) |
| `vip_init failed` / undefined `vip_init` | `-DVIP_INIT_HAS_SIZE=ON` ([1.8](#18-shared-cmake-npu)) |
| `pegasus not on PATH` | Set `ACUITY_PATH` and `VIV_SDK` on Linux/WSL; no sudo |
| ACUITY fails on ONNX | You exported e2e. Use `export_npu.py` (`*_6.onnx`) |
| Missing `.nb` after `git pull` | Files are gitignored — scp from the PC ([3.4](#34-copy-to-the-board)) |
| `yolo26_npu` given an `.onnx` | Convert first ([3](#3-convert-models-to-npu)) |
| NPU boxes in the wrong place / huge / garbage classes / thousands of boxes | **Reconvert** (old NBGs had `add_preproc_node: false`). `git pull` and rebuild `yolo26_npu`. Prefer FP16. Check `add_preproc_node: true` / `IMAGE_RGB`. |
| Car labeled as person / boat, near-full-frame box | VIPLite output is CHW, not the reported HWC ([#23418](https://github.com/blakeblackshear/frigate/discussions/23418)). Rebuild; do not pass `--layout hwc`. |
| No detections / huge boxes (imgsz) | `--imgsz` must match export (640) |
| OpenCV window fails over SSH | `--no-show --save result.jpg` |
| Board OOM compiling | `--swap`, `cmake --build . -j1` |
| Board OOM running ONNX | `--threads 2`, `yolo26n` only |
| ONNX is slow | Normal on CPU. Use `yolo26_npu` + `.nb` |
| `/opt/yolov5/yolov5` works, this app does not | That demo is YOLOv5. YOLO26 needs this 6-head `.nb` |

---

## Checklists

**ONNX CPU**

1. Board: `./setup_onnx.sh` (`--swap` on 1 GB)
2. `cd Yolo26_ONNX && ./build.sh`
3. `./run_onnx.sh export/yolo26n_6.onnx export/calib/bus.jpg`
4. Time: `./bench_onnx.sh`

**NPU**

1. PC: `python export_npu.py --model yolo26n.pt --outdir export`
2. Linux/WSL: `./convert_npu.sh --onnx export/yolo26n_6.onnx` (FP16). INT8: add `--quant pcq`. Confirm `add_preproc_node: true` in the generated inputmeta. **Do not keep an old `.nb`.**
3. scp `*.nb` to the board (`export_nb/` is not in git)
4. Board: `./setup_board.sh` then `cd Yolo26_NPU && ./build.sh`
5. `./run_npu.sh export_nb/yolo26n_fp16.nb export/calib/bus.jpg`
6. Time: `./bench_npu.sh`
