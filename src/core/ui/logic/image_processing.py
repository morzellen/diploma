# src\core\ui\logic\image_processing.py
"""Модуль для обработки изображений."""

import gradio as gr
from typing import Generator, List, Tuple, Any
from core.utils.get_logger import logger
from .processing_state import ProcessingState


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
    processing_results = [None] * total_images

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