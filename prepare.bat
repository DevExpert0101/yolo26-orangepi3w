@echo off
setlocal
cd /d "%~dp0"

python -m pip install -U -r requirements-prepare.txt
if errorlevel 1 exit /b 1

python prepare.py --model yolo26n.pt --imgsz 640 --formats npu --outdir export %*
if errorlevel 1 exit /b 1

echo.
echo NPU ONNX is in export\yolo26n_6.onnx
echo On Linux/WSL with native ACUITY (no Docker):
echo   set ACUITY_PATH and VIV_SDK, then:
echo   convert_npu.sh --onnx export/yolo26n_6.onnx
echo Then copy the .nb to the Orange Pi and run:
echo   python infer.py --model export/yolo26n_6_pcq_a733.nb --source export/bus.jpg
endlocal
