# src/core/handlers/renaming_handler.py
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List

from core.generators.caption_generator import CaptionGenerator
from core.generators.translation_generator import TranslationGenerator
from core.handlers.base_handler import BaseHandler
from core.utils.get_logger import logger
from core.generators.exceptions import CaptionGenerationError, TranslationGenerationError


class RenamingHandler(BaseHandler):
    _caption_generator: CaptionGenerator = None
    _translation_generator: TranslationGenerator = None

    @classmethod
    def initialize_models(cls, caption_model: str, trans_model: str) -> None:
        """Инициализация моделей с логированием и обработкой ошибок"""
        try:
            logger.info(
                f"Инициализация моделей | Генерация подписей: {caption_model} | "
                f"Перевод: {trans_model}"
            )
            cls._caption_generator = CaptionGenerator(caption_model)
            cls._translation_generator = TranslationGenerator(trans_model)
            logger.success("Модели для переименования успешно инициализированы")
        except Exception as e:
            logger.critical(f"Ошибка инициализации моделей: {e}", exc_info=True)
            raise

    @classmethod
    def handle_photo_generator(cls, photo_tuple: tuple, caption_model: str, 
                              trans_model: str, check_cancelled: callable, 
                              target_language: str) -> Generator:
        """Обработка потока фотографий с улучшенным логированием"""
        logger.info("Запуск процесса переименования фотографий")
        try:
            cls.initialize_models(caption_model, trans_model)
            yield from super()._common_processing(photo_tuple, check_cancelled, target_language)
        except Exception as e:
            logger.error(f"Критическая ошибка в обработчике переименования: {e}", exc_info=True)
            raise

    @classmethod
    def _generate_object(cls, photo_path: str, photo_name: str) -> str:
        """Генерация подписи с обработкой ошибок"""
        logger.debug(f"Генерация подписи для {photo_name}")
        try:
            caption = cls._caption_generator.generate(photo_path, photo_name)
            logger.debug(f"Сгенерирована подпись для {photo_name}: {caption}")
            return caption
        except CaptionGenerationError as e:
            logger.error(f"Ошибка генерации подписи для {photo_name}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Непредвиденная ошибка генерации подписи: {e}", exc_info=True)
            raise CaptionGenerationError("Ошибка создания подписи") from e

    @classmethod
    def _translate_object(cls, generated_object: str, target_lang: str) -> str:
        """Перевод подписи с логированием"""
        logger.debug(f"Перевод подписи '{generated_object}' на {target_lang}")
        try:
            translation = cls._translation_generator.generate(generated_object, "en_XX", target_lang)
            logger.debug(f"Перевод завершен: '{generated_object}' -> '{translation}'")
            return translation
        except TranslationGenerationError as e:
            logger.error(f"Ошибка перевода '{generated_object}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Непредвиденная ошибка перевода: {e}", exc_info=True)
            raise TranslationGenerationError("Ошибка перевода подписи") from e

    @classmethod
    def save_photo(cls, new_names: List[str], photo_paths: List[str], save_dir: str) -> List[str]:
        """Сохранение переименованных файлов с улучшенным логированием"""
        logger.info(f"Сохранение {len(photo_paths)} файлов в {save_dir}")
        
        try:
            save_path = Path(save_dir)
            if len(photo_paths) > 1:
                save_path /= 'renamed_photos'
                logger.debug(f"Использование групповой директории: {save_path}")

            logger.debug("Проверка и создание директории для сохранения")
            save_path.mkdir(parents=True, exist_ok=True)
            
            photo_paths = [Path(p) for p in photo_paths]
            name_counter = defaultdict(int)
            results = []
            
            with ThreadPoolExecutor() as executor:
                futures = []
                logger.info(f"Запуск пула потоков для {len(photo_paths)} задач копирования")

                for new_name, path in zip(new_names, photo_paths):
                    try:
                        base_name = new_name.strip()
                        name_counter[base_name] += 1
                        count = name_counter[base_name]
                        
                        final_name = f"{base_name} {count}" if count > 1 else base_name
                        dest = save_path / f"{final_name}{path.suffix}"
                        
                        logger.debug(f"Подготовка к копированию: {path.name} -> {dest.name}")
                        futures.append(executor.submit(
                            cls._safe_copy_file,
                            path,
                            dest
                        ))
                    except Exception as e:
                        error_msg = f"Ошибка подготовки файла {path.name}: {e}"
                        logger.error(error_msg, exc_info=True)
                        results.append(error_msg)

                # Обработка результатов выполнения задач
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.debug(result)
                    except Exception as e:
                        error_msg = f"Ошибка выполнения задачи копирования: {e}"
                        logger.error(error_msg, exc_info=True)
                        results.append(error_msg)

            success_count = sum(1 for r in results if "Успешно" in r)
            logger.success(
                f"Сохранение завершено | Успешно: {success_count} | "
                f"С ошибками: {len(results) - success_count}"
            )
            return results

        except Exception as e:
            logger.critical(f"Критическая ошибка при сохранении: {e}", exc_info=True)
            raise