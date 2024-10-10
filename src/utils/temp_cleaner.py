import os
import shutil
import tempfile
from utils.logger import logger

def clear_temp():
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
