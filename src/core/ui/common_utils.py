# src/core/ui/common_utils.py
"""Модуль вспомогательных утилит для обработки изображений и управления состоянием."""

from functools import wraps
import os
from pathlib import Path
import sys
import shutil
import tempfile
import threading
from typing import Generator, List, Tuple, Any, Optional

import gradio as gr
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog

from core.utils.get_logger import logger


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


class ProcessingState:
    """Класс для thread-safe управления состоянием обработки (Singleton)."""
    
    _instance: Optional['ProcessingState'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> 'ProcessingState':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._is_cancelled = False
                cls._instance._state_lock = threading.RLock()
        return cls._instance
    
    @property
    def is_cancelled(self) -> bool:
        """Проверяет флаг отмены операции."""
        with self._state_lock:
            return self._is_cancelled
    
    def set_cancellation_state(self, value: bool) -> None:
        """Устанавливает флаг отмены операции."""
        with self._state_lock:
            self._is_cancelled = value


def cancel_operation() -> List:
    """Обрабатывает запрос на отмену текущей операции."""
    state = ProcessingState()
    state.set_cancellation_state(True)
    logger.info("Operation cancellation requested")
    gr.Info("Operation cancellation requested")
    return []


def process_images(
    images: List[Tuple[str, str]],
    processing_pipeline: callable,
    primary_model: Any,
    secondary_model: Any,
    target_language: str,
    start_message: str,
    finish_message: str,
    progress_tracker: gr.Progress = gr.Progress()
) -> Generator[Tuple[List, List], None, None]:
    """
    Универсальный обработчик изображений с использованием генератора.
    
    Args:
        images: Список изображений для обработки
        processing_pipeline: Генератор обработки изображений
        primary_model: Основная модель обработки
        secondary_model: Вспомогательная модель обработки
        target_language: Целевой язык для перевода
        start_message: Стартовое сообщение
        finish_message: Финишное сообщение
        progress_tracker: Трекер прогресса
        
    Yields:
        Кортеж (текущий прогресс, исходные изображения)
    """
    state = ProcessingState()
    state.set_cancellation_state(False)

    if not images:
        logger.warning("No images uploaded for processing")
        gr.Warning("No images uploaded for processing")
        yield [], []
        return

    total_images = len(images)
    processing_results = [None] * total_images  # Трекер статусов обработки

    try:
        progress_tracker(0, desc=start_message)
        gr.Info(start_message)

        processing_generator = processing_pipeline(
            images,
            primary_model,
            secondary_model,
            lambda: state.is_cancelled,
            target_language
        )

        for result in processing_generator:
            if state.is_cancelled:
                gr.Warning("Operation cancelled by user")
                yield _generate_progress_report(processing_results), images
                return

            if _is_valid_processing_result(result, total_images):
                image_idx, processed_data = result
                processing_results[image_idx] = processed_data

            yield _generate_progress_report(processing_results), images

        if not state.is_cancelled:
            progress_tracker(1.0, desc="Completed!")
            gr.Info(finish_message)
            yield _generate_progress_report(processing_results), images

    except Exception as error:
        logger.error(f"Critical error: {str(error)}", exc_info=True)
        gr.Warning(f"Processing error: {str(error)}")
        yield _generate_progress_report(processing_results), images


def _generate_progress_report(processing_status: List) -> List:
    """Генерирует отчет о текущем прогрессе обработки."""
    return [
        [idx + 1, value[1] if isinstance(value, tuple) else value]
        for idx, value in enumerate(processing_status)
        if value is not None
    ]


def _is_valid_processing_result(result: Any, max_index: int) -> bool:
    """Проверяет валидность результата обработки."""
    return (
        isinstance(result, tuple) and 
        len(result) == 2 and 
        0 <= result[0] < max_index
    )


def create_processing_tab(handler_class, process_name):
    def decorator(func):
        @wraps(func)
        def wrapper(photo_tuple, model_param1, model_param2, tgt_lang_str):
            for progress, result in process_images(
                photo_tuple,
                handler_class.handle_photo_generator,
                model_param1,
                model_param2,
                tgt_lang_str,
                f"Начало процесса {process_name}...",
                f"Процесс {process_name} завершен"
            ):
                yield progress, result
        return wrapper
    return decorator


def save_processing_results(
    metadata: gr.DataFrame,
    original_images: List[Tuple[str, str]],
    output_dir: str,
    metadata_column: str,
    default_prefix: str,
    save_handler: callable,
    progress_tracker: gr.Progress = gr.Progress()
) -> None:
    """
    Сохраняет результаты обработки изображений.
    
    Args:
        metadata: Метаданные для сохранения
        original_images: Оригинальные изображения
        output_dir: Целевая директория
        metadata_column: Название колонки с метаданными
        default_prefix: Префикс по умолчанию
        save_handler: Обработчик сохранения
        progress_tracker: Трекер прогресса
    """
    if not original_images or metadata.empty:
        logger.warning("No data to save")
        gr.Warning("No data to save")
        return

    try:
        progress_tracker(0, desc="Validating data...")
        
        # Извлечение и валидация индексов
        indices = pd.to_numeric(metadata['№'], errors='coerce')
        valid_indices = indices.fillna(0).astype(int) - 1
        is_valid = (valid_indices >= 0) & (valid_indices < len(original_images))
        
        # Подготовка данных для сохранения
        filtered_indices = valid_indices[is_valid]
        filenames = metadata[metadata_column].astype(str).str.strip()
        processed_names = filenames[is_valid]

        # Генерация имен файлов
        default_names = [
            f"{default_prefix}_{idx+1}" if default_prefix else str(idx+1)
            for idx in filtered_indices
        ]
        final_names = processed_names.where(processed_names.ne(""), default_names)
        
        # Формирование путей
        image_paths = [
            Path(original_images[idx][0])
            if isinstance(original_images[idx], tuple)
            else Path(original_images[idx])
            for idx in filtered_indices
        ]
        
        progress_tracker(0.3, desc="Saving files...")
        
        os.makedirs("results", exist_ok=True) if not os.path.exists("results") else None

        progress_tracker(0.7, desc="Preparing paths...")

        str_paths = [str(p) for p in image_paths]
        save_results = save_handler(final_names.tolist(), str_paths, output_dir)

        # Обработка результатов
        success_count = sum(1 for res in save_results if "успешно" in res.lower())
        
        if success_count == len(image_paths):
            gr.Info(f"Successfully saved {success_count} files")
        else:
            gr.Warning(f"Saved {success_count} files")

        progress_tracker(1.0, desc="Operation completed!")

    except Exception as error:
        logger.error(f"Save error: {str(error)}", exc_info=True)
        gr.Warning(f"Save error: {str(error)}")
        progress_tracker(1.0, desc="Error occurred!")
        

def create_save_decorator(handler_class, column_name, default_prefix):
    def decorator(func):
        @wraps(func)
        def wrapper(df_data, photo_tuple, save_dir):
            save_processing_results(
                df_data,
                photo_tuple,
                save_dir,
                column_name,
                default_prefix,
                handler_class.save_photo
            )
            return photo_tuple, df_data
        return wrapper
    return decorator


def clear_temporary_data() -> List:
    """Очищает временные данные и сбрасыет состояние UI."""
    temp_directory = tempfile.gettempdir()
    gradio_cache = os.path.join(temp_directory, "gradio")
    
    logger.info(f"Cleaning temporary directory: {gradio_cache}")
    
    if os.path.exists(gradio_cache):
        try:
            shutil.rmtree(gradio_cache, ignore_errors=True)
            logger.info("Temporary data cleared successfully")
            gr.Info("Successfully!")
        except OSError as error:
            logger.error(f"Cleanup error: {str(error)}")

    return []


