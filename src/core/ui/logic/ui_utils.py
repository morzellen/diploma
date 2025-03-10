# src\core\ui\logic\ui_utils.py
"""Модуль утилит для работы с пользовательским интерфейсом."""

from typing import List, Tuple
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os
import gradio as gr


def select_directory(dialog_title: str, initial_dir: str = "") -> str:
    """
    Отображает диалог выбора директории.
    
    Args:
        dialog_title: Заголовок диалогового окна
        initial_dir: Начальная директория для выбора
        
    Returns:
        Выбранный путь или initial_dir при отмене
    """
    app = QApplication(sys.argv)
    selected_path = QFileDialog.getExistingDirectory(None, dialog_title)
    app.quit()
    return selected_path if selected_path else initial_dir


def initialize_photo_gallery(uploaded_photos: List[Tuple[str, str]]) -> Tuple[List, List]:
    """
    Инициализирует галерею изображений и таблицу метаданных.
    
    Args:
        uploaded_photos: Список кортежей (путь, описание) загруженных фото
        
    Returns:
        Кортеж (список фото для галереи, данные для таблицы)
    """
    if not uploaded_photos:
        return [], []

    gallery_items = [
        (photo[0], f'{idx}) {os.path.basename(photo[0])}')
        for idx, photo in enumerate(uploaded_photos, start=1)
    ]
    
    dataframe_entries = [[idx, ""] for idx in range(1, len(uploaded_photos) + 1)]
    
    return gallery_items, dataframe_entries


def update_button_states(is_processing: bool, 
                        process_button: gr.components.Component, 
                        cancel_button: gr.components.Component) -> dict:
    """
    Обновляет состояние кнопок управления процессом.
    
    Args:
        is_processing: Флаг активности процесса
        process_button: Основная кнопка запуска
        cancel_button: Кнопка отмены
        
    Returns:
        Словарь с новыми состояниями компонентов
    """
    return {
        process_button: gr.update(interactive=not is_processing),
        cancel_button: gr.update(visible=is_processing)
    }