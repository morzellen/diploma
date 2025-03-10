# src\core\utils\get_logger.py
import os
import logging
import sys
from datetime import datetime
from typing import Optional

# Конфигурация форматов логов
FILE_FORMAT = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s'
CONSOLE_FORMAT = '[%(levelname)s] %(message)s'

# Кастомный уровень логирования для успешных операций
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

class EnhancedLogger(logging.Logger):
    """Кастомный логгер с дополнительными функциями"""
    
    def success(self, msg, *args, **kwargs):
        """Логирование успешных операций"""
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Добавляем автоматическое логирование исключений"""
        super().critical(msg, *args, exc_info=True, **kwargs)

logging.setLoggerClass(EnhancedLogger)

def setup_logger(name: str = 'main') -> EnhancedLogger:
    """Инициализация и настройка логгера"""
    
    # Создание основной директории для логов
    log_dir = "../logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError as e:
        sys.stderr.write(f"Ошибка доступа к директории логов: {e}\n")
        raise

    # Инициализация логгера
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Настройка обработчика для файлов с timestamp
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    file_handler.setLevel(logging.DEBUG)

    # Настройка цветного вывода в консоль
    class ColorFormatter(logging.Formatter):
        """Форматирование с цветовой разметкой для терминала"""
        COLORS = {
            logging.DEBUG: "\033[37m",     # Белый
            logging.INFO: "\033[94m",       # Синий
            SUCCESS_LEVEL: "\033[92m",      # Зеленый
            logging.WARNING: "\033[93m",    # Желтый
            logging.ERROR: "\033[91m",      # Красный
            logging.CRITICAL: "\033[41m"    # Красный фон
        }
        RESET = "\033[0m"

        def format(self, record):
            color = self.COLORS.get(record.levelno, "")
            fmt = f"{color}{CONSOLE_FORMAT}{self.RESET}"
            return logging.Formatter(fmt).format(record)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    console_handler.setLevel(logging.INFO)

    # Добавление обработчиков
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Отключение дублирования логов
    logger.propagate = False

    return logger

# Инициализация глобального логгера
logger = setup_logger()
