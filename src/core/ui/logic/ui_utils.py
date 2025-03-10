# src\core\ui\logic\ui_utils.py
"""Модуль утилит для работы с пользовательским интерфейсом."""

from typing import List, Tuple, Dict, Any
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os
import gradio as gr
from core.utils.get_logger import logger


def select_directory(dialog_title: str, initial_dir: str = "") -> str:
    """
    Отображает диалог выбора директории с обработкой ошибок.
    """
    logger.info(f"Запрос выбора директории: '{dialog_title}'")
    selected_path = initial_dir
    app = None
    
    try:
        app = QApplication(sys.argv)
        logger.debug("Инициализировано QApplication")
        
        selected_path = QFileDialog.getExistingDirectory(None, dialog_title, initial_dir)
        status = "успешно" if selected_path else "отменено"
        logger.info(f"Выбор директории {status}: {selected_path or initial_dir}")
        
    except Exception as e:
        logger.error(
            f"Ошибка при выборе директории: {e}",
            exc_info=True
        )
        gr.Warning(f"Ошибка выбора директории: {str(e)}")
        
    finally:
        if app:
            app.quit()
            logger.debug("QApplication завершил работу")
            
    return selected_path if selected_path else initial_dir


def initialize_photo_gallery(uploaded_photos: List[Tuple[str, str]]) -> Tuple[List, List]:
    """
    Инициализирует галерею изображений с валидацией данных.
    """
    logger.info("Инициализация галереи изображений")
    
    if not uploaded_photos:
        logger.warning("Попытка инициализации пустой галереи")
        return [], []

    gallery_items = []
    dataframe_entries = []
    error_count = 0

    try:
        for idx, photo in enumerate(uploaded_photos, start=1):
            try:
                if not isinstance(photo, tuple) or len(photo) < 1:
                    raise ValueError("Некорректный формат элемента")
                    
                photo_path = photo[0]
                if not os.path.exists(photo_path):
                    raise FileNotFoundError(f"Файл не найден: {photo_path}")
                
                item = (photo_path, f'{idx}) {os.path.basename(photo_path)}')
                gallery_items.append(item)
                dataframe_entries.append([idx, ""])
                
                logger.debug(f"Добавлено изображение #{idx}: {photo_path}")
                
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Ошибка обработки элемента #{idx}: {e}",
                    exc_info=True
                )

        logger.info(
            f"Галерея инициализирована | Успешно: {len(gallery_items)} | "
            f"Ошибки: {error_count}"
        )
        
    except Exception as e:
        logger.error(
            f"Критическая ошибка инициализации галереи: {e}",
            exc_info=True
        )
        return [], []

    return gallery_items, dataframe_entries


def update_button_states(is_processing: bool, 
                        process_button: gr.components.Component, 
                        cancel_button: gr.components.Component) -> Dict[str, Any]:
    """
    Обновляет состояние кнопок с валидацией входных данных.
    """
    try:
        if not isinstance(is_processing, bool):
            raise TypeError(f"Некорректный тип флага обработки: {type(is_processing)}")
            
        logger.debug(
            f"Обновление состояния кнопок | Обработка: {is_processing} | "
            f"Кнопка процесса: {'активна' if not is_processing else 'неактивна'} | "
            f"Кнопка отмены: {'видима' if is_processing else 'скрыта'}"
        )
        
        return {
            process_button: gr.update(interactive=not is_processing),
            cancel_button: gr.update(visible=is_processing)
        }
        
    except Exception as e:
        logger.error(
            f"Ошибка обновления состояния кнопок: {e}",
            exc_info=True
        )
        return {}
    