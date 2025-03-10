# src\core\ui\logic\cancellation.py
"""Модуль для обработки отмены операций с расширенным логированием."""

from typing import List
import gradio as gr
from core.utils.get_logger import logger
from .processing_state import ProcessingState


def cancel_operation() -> List:
    """Обрабатывает запрос на отмену текущей операции."""
    try:
        state = ProcessingState()
        
        if not state.is_cancelled:
            logger.warning("Инициирована отмена операции пользователем")
            state.set_cancellation_state(True)
            logger.debug("Флаг отмены успешно установлен")
            try:
                gr.Info("Operation cancellation requested. Please wait...")
            except Exception as e:
                logger.error(f"Ошибка отображения уведомления Gradio: {e}", exc_info=True)
        else:
            logger.debug("Повторный запрос отмены уже активной операции")
            
        return []

    except Exception as e:
        logger.critical(
            f"Критическая ошибка при обработке отмены: {e}", 
            exc_info=True
        )
        try:
            gr.Warning("Failed to process cancellation request")
        except Exception as gradio_error:
            logger.error(f"Ошибка отображения предупреждения Gradio: {gradio_error}")
        return []
    