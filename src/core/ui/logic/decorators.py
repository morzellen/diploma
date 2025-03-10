# src\core\ui\logic\decorators.py
"""Модуль декораторов с расширенным логированием и обработкой ошибок."""

from functools import wraps
from typing import Callable, Any
from core.utils.get_logger import logger
from .image_processing import process_images
from .data_management import save_processing_results


def create_processing_tab(handler_class: Any, process_name: str) -> Callable:
    """Декоратор для создания вкладки обработки с логированием."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(photo_tuple: tuple, model_param1: str, model_param2: str, tgt_lang_str: str) -> Any:
            logger.info(
                f"Запуск процесса '{process_name}' | "
                f"Модели: {model_param1}/{model_param2} | "
                f"Язык: {tgt_lang_str}"
            )
            
            try:
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
                    
                logger.success(f"Процесс '{process_name}' успешно завершен")

            except Exception as e:
                logger.critical(
                    f"Критическая ошибка в процессе '{process_name}': {e}",
                    exc_info=True
                )
                raise
                
            finally:
                logger.debug(f"Завершение обработки для '{process_name}'")

        return wrapper
    return decorator


def create_save_decorator(handler_class: Any, column_name: str, default_prefix: str) -> Callable:
    """Декоратор для операций сохранения с проверкой данных."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df_data: Any, photo_tuple: list, save_dir: str) -> Any:
            logger.info(
                f"Сохранение результатов | Колонка: {column_name} | "
                f"Префикс: {default_prefix} | Директория: {save_dir}"
            )
            
            try:
                if not hasattr(handler_class, 'save_photo'):
                    error_msg = f"Класс {handler_class.__name__} не содержит метод save_photo"
                    logger.error(error_msg)
                    raise AttributeError(error_msg)

                save_processing_results(
                    df_data,
                    photo_tuple,
                    save_dir,
                    column_name,
                    default_prefix,
                    handler_class.save_photo
                )
                
                logger.success("Результаты сохранения успешно обработаны")
                return photo_tuple, df_data
                
            except Exception as e:
                logger.error(
                    f"Ошибка сохранения результатов: {e}",
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator
