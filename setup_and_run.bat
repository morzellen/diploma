@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Проверка и установка Python 3.11.8
python --version 2>&1 | find "3.11.8" > nul
if %errorlevel% neq 0 (
    echo Установите Python 3.11.8 с официального сайта: https://www.python.org/downloads/release/python-3118/
    pause
    exit /b 1
)

:: Создание виртуального окружения
if not exist "venv\" (
    echo Создание виртуального окружения...
    python -m venv venv
)

:: Активация venv
call venv\Scripts\activate.bat

:: Установка библиотек
echo Проверка зависимостей...
pip install -r installed_libs.txt

:: Проверка CUDA
echo Проверка поддержки CUDA...
nvidia-smi 2>nul
if %errorlevel% neq 0 (
    echo ВНИМАНИЕ: CUDA-совместимая видеокарта не обнаружена или драйверы не установлены
    echo Рекомендуется установить CUDA Toolkit с https://developer.nvidia.com/cuda-downloads
    timeout /t 5
)

:: Запуск приложения
echo Запуск приложения...
cd src
python main.py
cd ..

pause