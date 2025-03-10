import os
import logging
import sys
from datetime import datetime

# Полный формат для файла
FORMAT = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s'
# Упрощённый формат для консоли
SIMPLE_FORMAT = '[%(levelname)s] %(message)s'

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