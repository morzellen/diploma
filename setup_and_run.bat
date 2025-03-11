@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: �������� � ��������� venv
if not exist "venv\" (
    echo �������� ������������ ���������...
    python -m venv venv
)
call venv\Scripts\activate.bat

:: ���������� pip � ��������� ������������
python -m pip install --upgrade pip --no-cache-dir
echo ��������� ���������...
pip install -r installed_libs.txt --no-cache-dir
if %errorlevel% neq 0 (
    echo ������ ��������� ������������! ��������� ���� installed_libs.txt
    pause
    exit /b 1
)

echo ��������� torch...
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo ������ ��������� torch! ��������� ���� installed_libs.txt
    pause
    exit /b 1
)

:: ������ ����������
cd src
python main.py
if %errorlevel% neq 0 (
    echo ������ ������� ����������!
    cd ..
    pause
    exit /b 1
)
cd ..
pause
