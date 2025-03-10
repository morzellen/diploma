# src/core/handlers/classification_handler.py
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List

from core.generators.exceptions import SegmentationGenerationError, TranslationGenerationError
from core.generators.segment_generator import SegmentGenerator
from core.generators.translation_generator import TranslationGenerator
from core.handlers.base_handler import BaseHandler

from core.utils.get_logger import logger

class ClassificationHandler(BaseHandler):
    _segment_generator: SegmentGenerator = None
    _translation_generator: TranslationGenerator = None

    @classmethod
    def initialize_models(cls, seg_model: str, trans_model: str) -> None:
        """Инициализация моделей с обработкой ошибок"""
        try:
            logger.info(
                f"Инициализация моделей | Сегментация: {seg_model} | Перевод: {trans_model}"
            )
            cls._segment_generator = SegmentGenerator(seg_model)
            cls._translation_generator = TranslationGenerator(trans_model)
            logger.success("Модели успешно инициализированы")
        except Exception as e:
            logger.critical(f"Ошибка инициализации моделей: {e}", exc_info=True)
            raise

    @classmethod
    def handle_photo_generator(cls, photo_tuple: tuple, seg_model: str, trans_model: str, 
                              check_cancelled: callable, target_language: str) -> Generator:
        """Обработка фотографий с логированием этапов"""
        logger.info("Запуск обработки изображений для классификации")
        try:
            cls.initialize_models(seg_model, trans_model)
            yield from super()._common_processing(photo_tuple, check_cancelled, target_language)
        except Exception as e:
            logger.error(f"Критическая ошибка в основном цикле обработки: {e}", exc_info=True)
            raise

    @classmethod
    def _generate_object(cls, photo_path: str, photo_name: str) -> str:
        """Генерация объекта сегментации с обработкой ошибок"""
        logger.debug(f"Генерация сегмента для {photo_name}")
        try:
            result = cls._segment_generator.generate(photo_path, photo_name)
            logger.debug(f"Результат сегментации {photo_name}: {result}")
            return result
        except SegmentationGenerationError as e:
            logger.error(f"Ошибка сегментации {photo_name}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Непредвиденная ошибка сегментации {photo_name}: {e}", exc_info=True)
            raise SegmentationGenerationError("Ошибка генерации сегмента") from e

    @classmethod
    def _translate_object(cls, generated_object: str, target_lang: str) -> str:
        """Перевод объекта с обработкой ошибок"""
        logger.debug(f"Перевод объекта '{generated_object}' на {target_lang}")
        try:
            result = cls._translation_generator.generate(generated_object, "en_XX", target_lang)
            logger.debug(f"Результат перевода: {result}")
            return result
        except TranslationGenerationError as e:
            logger.error(f"Ошибка перевода '{generated_object}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Непредвиденная ошибка перевода: {e}", exc_info=True)
            raise TranslationGenerationError("Ошибка перевода объекта") from e

    @classmethod
    def save_photo(cls, class_names: List[str], photo_paths: List[str], save_dir: str) -> List[str]:
        """Сохранение классифицированных фотографий с улучшенным логированием"""
        logger.info(f"Сохранение {len(photo_paths)} фото в директорию: {save_dir}")
        save_path = Path(save_dir) / 'classified_photos'
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория для сохранения: {save_path}")
        except Exception as e:
            logger.error(f"Ошибка создания директории {save_path}: {e}", exc_info=True)
            raise

        photo_paths = [Path(p) for p in photo_paths]
        class_files = defaultdict(list)
        
        # Группировка по классам
        for class_name, photo_path in zip(class_names, photo_paths):
            class_files[class_name].append(photo_path)
            logger.debug(f"Классификация: {photo_path.name} -> {class_name}")

        results = []
        try:
            with ThreadPoolExecutor() as executor:
                futures = []
                logger.info(f"Запуск пула потоков для сохранения {len(photo_paths)} файлов")
                
                for class_name, paths in class_files.items():
                    class_dir = save_path / class_name
                    try:
                        class_dir.mkdir(exist_ok=True)
                        logger.debug(f"Создана директория класса: {class_name}")
                    except Exception as e:
                        logger.error(f"Ошибка создания директории {class_name}: {e}", exc_info=True)
                        continue

                    for path in paths:
                        dest = class_dir / path.name
                        futures.append(executor.submit(
                            cls._safe_copy_file, 
                            path, 
                            dest
                        ))
                        logger.debug(f"Добавлена задача копирования: {path.name}")

                # Обработка результатов
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.debug(result)
                    except Exception as e:
                        error_msg = f"Ошибка выполнения задачи: {e}"
                        logger.error(error_msg, exc_info=True)
                        results.append(error_msg)

            logger.success(f"Успешно сохранено {len(results)} файлов")
            return results

        except Exception as e:
            logger.critical(f"Критическая ошибка при сохранении: {e}", exc_info=True)
            raise
