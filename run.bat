@echo off
cd /d "%~dp0"
title 车牌识别系统启动器

:: 手动指定正确的 Python 解释器路径
set PYTHON_EXE=D:\新建文件夹\python\python.exe

:: 检查文件是否存在
if not exist "%PYTHON_EXE%" (
    echo [错误] 找不到 Python 解释器: %PYTHON_EXE%
    echo 请修改 run.bat 中的 PYTHON_EXE 路径。
    pause
    exit /b
)

echo [信息] 使用 Python: %PYTHON_EXE%

echo [1/2] 检查/安装依赖...
%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install streamlit hyperlpr3 openai opencv-python pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接。
    pause
    exit /b
)

echo [2/2] 启动程序...
%PYTHON_EXE% -m streamlit run app.py
pause