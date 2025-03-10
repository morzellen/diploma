# src\core\ui\logic\data_management.py
"""Модуль для управления данными и сохранения результатов с расширенным логированием."""

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
    Сохраняет результаты обработки изображений с улучшенным логированием.
    """
    try:
        logger.info("Начало сохранения результатов обработки")
        
        if not original_images or metadata.empty:
            logger.warning("Попытка сохранения без данных: отсутствуют изображения или метаданные")
            gr.Warning("No data to save")
            return

        progress_tracker(0, desc="Validating data...")
        logger.debug("Валидация входных данных")

        # Обработка индексов
        try:
            indices = pd.to_numeric(metadata['№'], errors='coerce')
            valid_indices = indices.fillna(0).astype(int) - 1
            is_valid = (valid_indices >= 0) & (valid_indices < len(original_images))
            
            invalid_count = len(is_valid) - sum(is_valid)
            if invalid_count > 0:
                logger.warning(f"Найдено {invalid_count} невалидных индексов")

            filtered_indices = valid_indices[is_valid]
        except KeyError as ke:
            logger.error(f"Отсутствует обязательная колонка '№': {ke}", exc_info=True)
            gr.Warning("Missing required column '№' in metadata")
            return

        # Обработка имен файлов
        try:
            filenames = metadata[metadata_column].astype(str).str.strip()
            processed_names = filenames[is_valid]
            
            empty_names = processed_names.empty or processed_names.str.len() == 0
            if empty_names.any():
                logger.debug(f"Найдено {empty_names.sum()} пустых имен, используются значения по умолчанию")

            default_names = [
                f"{default_prefix}_{idx+1}" if default_prefix else str(idx+1)
                for idx in filtered_indices
            ]
            final_names = processed_names.where(processed_names.ne(""), default_names)
        except Exception as e:
            logger.error(f"Ошибка обработки имен файлов: {e}", exc_info=True)
            gr.Warning("Error processing filenames")
            return

        # Подготовка путей
        try:
            image_paths = []
            for idx in filtered_indices:
                try:
                    path = Path(original_images[idx][0]) if isinstance(original_images[idx], tuple) else Path(original_images[idx])
                    image_paths.append(path)
                    logger.debug(f"Обработка пути: {path}")
                except (IndexError, TypeError) as te:
                    logger.error(f"Ошибка доступа к изображению по индексу {idx}: {te}")
                    continue

            if not image_paths:
                logger.error("Нет валидных путей к изображениям для сохранения")
                gr.Warning("No valid image paths found")
                return
        except Exception as e:
            logger.error(f"Критическая ошибка подготовки путей: {e}", exc_info=True)
            gr.Warning("Path preparation error")
            return

        # Сохранение файлов
        try:
            progress_tracker(0.3, desc="Saving files...")
            logger.info(f"Сохранение {len(image_paths)} файлов в {output_dir}")

            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Создана выходная директория: {output_dir}")

            str_paths = [str(p) for p in image_paths]
            save_results = save_handler(final_names.tolist(), str_paths, output_dir)

            success_count = sum(1 for res in save_results if "успешно" in res.lower())
            error_count = len(save_results) - success_count
            
            logger.info(f"Результаты сохранения: Успешно - {success_count}, Ошибок - {error_count}")

            if error_count > 0:
                error_examples = [res for res in save_results if "ошибка" in res.lower()][:3]
                logger.warning(f"Примеры ошибок при сохранении: {error_examples}")

            if success_count == len(image_paths):
                gr.Info(f"Successfully saved {success_count} files")
            else:
                gr.Warning(f"Saved {success_count} files with {error_count} errors")

            progress_tracker(1.0, desc="Operation completed!")
            logger.success("Процесс сохранения завершен")

        except Exception as e:
            logger.error(f"Ошибка в процессе сохранения файлов: {e}", exc_info=True)
            gr.Warning("File saving error")
            progress_tracker(1.0, desc="Error occurred!")

    except Exception as error:
        logger.critical(
            f"Критическая ошибка при сохранении результатов: {error}",
            exc_info=True
        )
        gr.Warning(f"Save error: {str(error)}")
        progress_tracker(1.0, desc="Error occurred!")


def clear_temporary_data() -> List:
    """Очищает временные данные с улучшенным логированием."""
    try:
        temp_directory = tempfile.gettempdir()
        gradio_cache = os.path.join(temp_directory, "gradio")
        
        logger.info(f"Очистка временных данных в директории: {gradio_cache}")

        if not os.path.exists(gradio_cache):
            logger.debug("Временная директория не найдена, пропуск очистки")
            return []

        try:
            shutil.rmtree(gradio_cache, ignore_errors=True)
            logger.success("Временные данные успешно удалены")
            
            if os.path.exists(gradio_cache):
                remaining = len(os.listdir(gradio_cache))
                logger.warning(f"Не удалось удалить {remaining} элементов в кэше")
            
            gr.Info("Temporary data cleared successfully")
        except OSError as error:
            logger.error(
                f"Ошибка очистки временных данных: {error}",
                exc_info=True
            )
            gr.Warning("Failed to clear temporary data")

        return []

    except Exception as e:
        logger.critical(
            f"Критическая ошибка при очистке временных данных: {e}",
            exc_info=True
        )
        return []
    