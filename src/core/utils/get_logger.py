# src\core\utils\get_logger.py
import os
import logging
import sys
from datetime import datetime

# Полный формат для файла
FORMAT = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s'
# Упрощённый формат для консоли
SIMPLE_FORMAT = '[%(levelname)s] %(message)s'

# Добавляем кастомный уровень SUCCESS
SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

class CustomLogger(logging.getLoggerClass()):
    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)

logging.setLoggerClass(CustomLogger)

# Создаём логгер
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)  # Минимальный уровень логирования

# Создаём папку для логов
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

# Файловый обработчик (пишет все сообщения от DEBUG и выше)
file_handler = logging.FileHandler(
    filename=os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(FORMAT))
file_handler.setLevel(logging.DEBUG)

# Консольный обработчик (выводит в терминал)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))
stream_handler.setLevel(logging.DEBUG)

# Добавляем обработчики к логгеру
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Отключаем propagation для избежания дублирования
logger.propagate = False

# Добавляем стилизованный вывод для уровня SUCCESS в консольный обработчик
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: "[DEBUG] %(message)s",
        logging.INFO: "\033[94m[INFO]\033[0m %(message)s",
        SUCCESS: "\033[92m[SUCCESS]\033[0m %(message)s",
        logging.WARNING: "\033[93m[WARNING]\033[0m %(message)s",
        logging.ERROR: "\033[91m[ERROR]\033[0m %(message)s",
        logging.CRITICAL: "\033[41m\033[97m[CRITICAL]\033[0m %(message)s"
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self._style._fmt)
        formatter = logging.Formatter(fmt)
        return formatter.format(record)

# Обновляем консольный обработчик
stream_handler.setFormatter(ColoredFormatter())
