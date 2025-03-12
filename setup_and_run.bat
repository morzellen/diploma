@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM Определение корневой директории
set "ROOT_DIR=%~dp0"

REM Проверка наличия Python 3.9-3.11 в системе
set "PYTHON_EXE="
set "PYTHON_VERSION_OK=0"

where python >nul 2>&1
if %errorlevel% == 0 (
    for /f "delims=" %%a in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%a"
    for /f "tokens=2 delims= " %%b in ("!PYTHON_VERSION!") do (
        for /f "tokens=1-3 delims=." %%i in ("%%b") do (
            if "%%i" == "3" (
                if %%j geq 9 if %%j leq 11 (
                    set PYTHON_VERSION_OK=1
                    set PYTHON_EXE=python
                )
            )
        )
    )
)

REM Создание структуры папок
if not exist "%ROOT_DIR%venv\" mkdir "%ROOT_DIR%venv"

REM Логика установки Python
if !PYTHON_VERSION_OK! == 0 (
    echo Установка Python 3.10.11 в venv...
    
    if not exist "%ROOT_DIR%venv\python.exe" (
        echo Скачивание embedded Python...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile '%ROOT_DIR%python-embed.zip'"
        
        echo Распаковка в venv...
        powershell -Command "Expand-Archive -Path '%ROOT_DIR%python-embed.zip' -DestinationPath '%ROOT_DIR%venv'"
        del "%ROOT_DIR%python-embed.zip"
        
        echo Настройка окружения...
        del "%ROOT_DIR%venv\python._pth" 2>nul
        echo. > "%ROOT_DIR%venv\python._pth"
        
        echo Установка pip...
        powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%ROOT_DIR%get-pip.py'"
        "%ROOT_DIR%venv\python.exe" "%ROOT_DIR%get-pip.py"
        del "%ROOT_DIR%get-pip.py"
    )
    set "PYTHON_EXE=%ROOT_DIR%venv\python.exe"
) else (
    echo Обнаружен подходящий Python. Создание venv...
    python -m venv "%ROOT_DIR%venv"
    set "PYTHON_EXE=%ROOT_DIR%venv\Scripts\python.exe"
)

REM Активация виртуального окружения
set "VIRTUAL_ENV=%ROOT_DIR%venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%VIRTUAL_ENV%;%PATH%"

REM Проверка активации
echo Проверка окружения...
%PYTHON_EXE% -c "import sys; print(sys.executable)" | find "%ROOT_DIR%venv" >nul
if %errorlevel% neq 0 (
    echo Ошибка активации виртуального окружения!
    pause
    exit /b 1
)

REM Обновление pip
echo Обновление pip...
%PYTHON_EXE% -m pip install --upgrade pip==24.0
call :CHECK_ERROR "Ошибка обновления pip"

REM Проверка CUDA
set "CUDA_AVAILABLE=0"
where nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    set CUDA_AVAILABLE=1
)

REM Установка torch
if !CUDA_AVAILABLE! == 1 (
    echo Установка torch с поддержкой CUDA...
    set "TORCH_EXTRA=cu126"
    set "TORCH_URL=https://download.pytorch.org/whl/cu126"
) else (
    echo Установка torch для CPU...
    set "TORCH_EXTRA=cpu"
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"
)

echo Проверка установленных библиотек...
%PYTHON_EXE% -m pip show torch >nul 2>&1
if %errorlevel% neq 0 (
    echo Установка torch/torchvision...
    %PYTHON_EXE% -m pip cache purge
    %PYTHON_EXE% -m pip install --no-cache-dir torch==2.6.0+%TORCH_EXTRA% torchvision==0.21.0+%TORCH_EXTRA% --index-url %TORCH_URL%
    call :CHECK_ERROR "Ошибка установки torch/torchvision"
)

REM Установка зависимостей
if exist "requirements.txt" (
    echo Установка дополнительных зависимостей...
    %PYTHON_EXE% -m pip install -r requirements.txt
    call :CHECK_ERROR "Ошибка установки зависимостей"
)

REM Запуск приложения
echo Запуск приложения из виртуального окружения...
cd src
%PYTHON_EXE% main.py
call :CHECK_ERROR "Ошибка запуска приложения"
cd ..

pause
exit /b 0

:CHECK_ERROR
if %errorlevel% neq 0 (
    echo %~1
    pause
    exit /b 1
)
exit /b 0