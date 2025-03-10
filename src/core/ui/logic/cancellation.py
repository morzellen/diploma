"""Модуль для обработки отмены операций."""

from typing import List
import gradio as gr
from core.utils.get_logger import logger
from .processing_state import ProcessingState


def cancel_operation() -> List:
    """Обрабатывает запрос на отмену текущей операции."""
    state = ProcessingState()
    state.set_cancellation_state(True)
    logger.info("Operation cancellation requested")
    gr.Info("Operation cancellation requested")
    return []