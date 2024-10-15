import os
import shutil
import tempfile
import logging
from datetime import datetime
import torch

FORMAT = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s'

logging.basicConfig(format=FORMAT)

log_filename = f"../logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
handler = logging.FileHandler(log_filename, encoding='utf-8')
handler.setFormatter(logging.Formatter(FORMAT))

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def get_device():
    """Определяем доступное устройство для модели"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используемое устройство: {device}")
    return device

def clear_temp():
    """
    Функция для очистки временной директории.

    Эта функция получает путь к временной директории и имя папки, которую нужно удалить.
    Затем она формирует полный путь к папке и проверяет ее существование.
    Если папка существует, она пытается удалить ее с помощью shutil.rmtree.
    Если удаление прошло успешно, функция записывает сообщение в лог о том, что папка удалена.
    В противном случае функция записывает сообщение в лог об ошибке.
    Если папка не найдена, функция также записывает сообщение в лог.

    :return: None
    """
    # Получаем путь к папке в temp
    temp_dir = tempfile.gettempdir()

    # Указываем имя папки, которую нужно удалить
    folder_name = "gradio"

    logger.info(f"Начало очистки временной директории: {folder_name}")

    # Формируем полный путь к папке
    folder_path = os.path.join(temp_dir, folder_name)

    # Проверяем существование папки и удаляем ее
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            logger.info(f"Папка {folder_name} удалена.")
        except Exception as e:
            logger.error(f"Ошибка при удалении папки {folder_name}: {e}")
    else:
        logger.info(f"Папка {folder_name} не найдена.")