# src\core\ui\logic\image_processing.py
"""Модуль для обработки изображений с расширенным логированием."""

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
    """
    state = ProcessingState()
    state.set_cancellation_state(False)

    if not images:
        logger.warning("Не загружено изображений для обработки")
        gr.Warning("No images uploaded for processing")
        yield [], []
        return

    total_images = len(images)
    processing_results = [None] * total_images
    logger.info(
        f"Начало обработки {total_images} изображений | "
        f"Модели: {primary_model}/{secondary_model} | "
        f"Язык: {target_language}"
    )

    try:
        progress_tracker(0, desc=start_message)
        gr.Info(start_message)
        logger.info(f"Старт обработки: {start_message}")

        processing_generator = processing_pipeline(
            images,
            primary_model,
            secondary_model,
            lambda: state.is_cancelled,
            target_language
        )

        for result in processing_generator:
            if state.is_cancelled:
                logger.warning("Обработка прервана пользователем")
                gr.Warning("Operation cancelled by user")
                yield _generate_progress_report(processing_results), images
                return

            if _is_valid_processing_result(result, total_images):
                image_idx, processed_data = result
                processing_results[image_idx] = processed_data
                logger.debug(f"Обработано изображение #{image_idx + 1}/{total_images}")
            else:
                logger.warning(f"Некорректный результат обработки: {result}")

            yield _generate_progress_report(processing_results), images

        if not state.is_cancelled:
            progress_tracker(1.0, desc="Completed!")
            logger.success(finish_message)
            gr.Info(finish_message)
            yield _generate_progress_report(processing_results), images

    except Exception as error:
        logger.critical(
            f"Критическая ошибка обработки: {str(error)} | "
            f"Прогресс: {len([x for x in processing_results if x is not None])}/{total_images}",
            exc_info=True
        )
        gr.Warning(f"Processing error: {str(error)}")
        yield _generate_progress_report(processing_results), images
    finally:
        logger.debug("Завершение процесса обработки изображений")


def _generate_progress_report(processing_status: List) -> List:
    """Генерирует отчет о текущем прогрессе обработки."""
    logger.debug(f"Формирование отчёта для {len(processing_status)} элементов")
    return [
        [idx + 1, value[1] if isinstance(value, tuple) else value]
        for idx, value in enumerate(processing_status)
        if value is not None
    ]


def _is_valid_processing_result(result: Any, max_index: int) -> bool:
    """Проверяет валидность результата обработки."""
    is_valid = (
        isinstance(result, tuple) and 
        len(result) == 2 and 
        0 <= result[0] < max_index
    )
    if not is_valid:
        logger.warning(f"Невалидный результат обработки: {result}")
    return is_valid
