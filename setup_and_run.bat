@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Создание и активация venv
if not exist "venv\" (
    echo Создание виртуального окружения...
    python -m venv venv
)
call venv\Scripts\activate.bat

:: Обновление pip и установка зависимостей
python -m pip install --upgrade pip --no-cache-dir
echo Установка библиотек...
pip install -r installed_libs.txt --no-cache-dir
if %errorlevel% neq 0 (
    echo Ошибка установки зависимостей! Проверьте файл installed_libs.txt
    pause
    exit /b 1
)

echo Установка torch...
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo Ошибка установки torch! Проверьте файл installed_libs.txt
    pause
    exit /b 1
)

:: Запуск приложения
cd src
python main.py
if %errorlevel% neq 0 (
    echo Ошибка запуска приложения!
    cd ..
    pause
    exit /b 1
)
cd ..
pause
