import os
import shutil
import tempfile

from core.utils.get_logger import logger

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

    temp_dir = tempfile.gettempdir()  # Получаем путь к папке в temp

    folder_name = "gradio"  # Указываем имя папки, которую нужно удалить

    logger.info(f"Начало очистки временной директории: {folder_name}")

    folder_path = os.path.join(temp_dir, folder_name)  # Формируем полный путь к папке

    if os.path.exists(folder_path):  # Проверяем существование папки и удаляем ее
        shutil.rmtree(folder_path)
        logger.info(f"Папка {folder_name} удалена.")
    else:
        logger.info(f"Папка {folder_name} не найдена.")
