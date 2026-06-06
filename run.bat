@echo off
cd /d D:\license_plate_hw
echo 当前目录: %cd%
set PYTHON_EXE=D:\新建文件夹\python\python.exe
if not exist "%PYTHON_EXE%" (
    echo 找不到 Python: %PYTHON_EXE%
    pause
    exit /b
)
echo 使用 Python: %PYTHON_EXE%

echo 检查依赖...
%PYTHON_EXE% -m pip install pandas streamlit hyperlpr3 openai opencv-python pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo 依赖安装失败，请检查网络
    pause
    exit /b
)

echo 启动程序...
%PYTHON_EXE% -m streamlit run app.py
if errorlevel 1 (
    echo 启动失败，请检查 app.py
    pause
    exit /b
)
pause