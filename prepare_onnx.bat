@echo off
setlocal
cd /d "%~dp0"

python -m pip install -U -r requirements-prepare.txt
if errorlevel 1 exit /b 1

python prepare_onnx.py --model yolo26n.pt --imgsz 640 --outdir export_onnx %*
if errorlevel 1 exit /b 1

echo.
echo Standard ONNX is in export_onnx\
echo Copy that folder to the Orange Pi Zero 3W, then on the board:
echo   ./setup_onnx.sh
echo   cd Yolo26_ONNX ^&^& ./build.sh
echo   ./run_onnx.sh
endlocal
