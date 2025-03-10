# src\core\ui\logic\decorators.py
"""Модуль декораторов для обработки и сохранения данных."""

from functools import wraps
from .image_processing import process_images
from .data_management import save_processing_results


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
