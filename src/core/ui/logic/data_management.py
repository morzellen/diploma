# src\core\ui\logic\data_management.py
"""Модуль для управления данными и сохранения результатов."""

import os
from pathlib import Path
import shutil
import tempfile
from typing import List, Tuple
import gradio as gr
import pandas as pd
from core.utils.get_logger import logger


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
        
        indices = pd.to_numeric(metadata['№'], errors='coerce')
        valid_indices = indices.fillna(0).astype(int) - 1
        is_valid = (valid_indices >= 0) & (valid_indices < len(original_images))
        
        filtered_indices = valid_indices[is_valid]
        filenames = metadata[metadata_column].astype(str).str.strip()
        processed_names = filenames[is_valid]

        default_names = [
            f"{default_prefix}_{idx+1}" if default_prefix else str(idx+1)
            for idx in filtered_indices
        ]
        final_names = processed_names.where(processed_names.ne(""), default_names)
        
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


def clear_temporary_data() -> List:
    """Очищает временные данные и сбрасывает состояние UI."""
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