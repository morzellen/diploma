# src/core/handlers/base_handler.py
from abc import ABC, abstractmethod
import shutil
from typing import Generator
from pathlib import Path
from core.utils.get_logger import logger


class BaseHandler(ABC):
    @classmethod
    @abstractmethod
    def handle_photo_generator(cls, *args) -> Generator:
        pass

    @classmethod
    @abstractmethod
    def save_photo(cls, *args) -> list:
        pass

    @classmethod
    def _common_processing(cls, photo_tuple, check_cancelled, target_lang):
        logger.info(f"Начало обработки пакета из {len(photo_tuple)} изображений")
        try:
            for i, item in enumerate(photo_tuple):
                if check_cancelled and check_cancelled():
                    logger.warning("Обработка прервана пользователем")
                    return
                try:
                    path = Path(item[0]) if isinstance(item, tuple) else Path(item)
                    logger.debug(f"Обработка элемента {i}: {path.name}")
                    yield from cls._process_single_photo(str(path), i, target_lang)
                except Exception as e:
                    logger.error(
                        f"Ошибка обработки элемента {i} ({path.name}): {e}",
                        exc_info=True
                    )
                    yield (i, f"Ошибка обработки: {str(e)}")
        finally:
            logger.info("Завершение обработки пакета изображений")

    @classmethod
    def _process_single_photo(cls, photo_path: str, index: int, target_lang: str) -> Generator:
        photo_name = Path(photo_path).name
        logger.info(f"Начало обработки изображения [{index}] {photo_name}")
        yield f"Обработка: {photo_name}"
        
        try:
            # Общая логика обработки
            logger.debug("Генерация основного объекта")
            original_object = cls._generate_object(photo_path, photo_name)
            logger.debug("Выполнение перевода объекта")
            translated = cls._translate_object(original_object, target_lang)
            
            logger.success(f"Успешная обработка изображения [{index}] {photo_name}")
            yield (index, (original_object, translated))
            
        except Exception as e:
            logger.error(
                f"Критическая ошибка обработки изображения [{index}] {photo_name}: {e}",
                exc_info=True
            )
            raise

    @abstractmethod
    def _generate_object(cls, photo_path: str, photo_name: str):
        """Генерация основного объекта (капшина/сегмента/др.)"""
        pass

    @abstractmethod
    def _translate_object(cls, generated_object, target_lang: str):
        """Трансляция сгенерированного объекта"""
        pass

    @staticmethod
    def _safe_copy_file(src: Path, dest: Path) -> str:
        """Потокобезопасное копирование файла с проверкой"""
        logger.debug(f"Попытка копирования: {src} -> {dest}")
        try:
            if not src.exists():
                error_msg = f"Файл не найден {src}"
                logger.error(error_msg)
                return error_msg
                
            if dest.exists():
                warn_msg = f"Файл уже существует: {dest}"
                logger.warning(warn_msg)
                return warn_msg
                
            logger.info(f"Копирование файла: {src.name}")
            shutil.copy(str(src), str(dest))
            
            success_msg = f"Успешно сохранён: {dest}"
            logger.success(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Ошибка копирования {src}: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg
