# src/core/handlers/base_handler.py
from abc import ABC, abstractmethod
import shutil
from typing import Generator
from pathlib import Path
from typing_extensions import Literal

class BaseHandler(ABC):

    @classmethod
    @abstractmethod
    def handle_photo_generator(cls, *args) -> Generator:
        pass


    @staticmethod
    # @abstractmethod
    def save_photo(*args) -> list:
        pass


    @classmethod
    def _common_processing(cls, photo_tuple, check_cancelled, target_lang):
        for i, item in enumerate(photo_tuple):
            if check_cancelled and check_cancelled():
                return
            try:
                path = Path(item[0]) if isinstance(item, tuple) else Path(item)
                yield from cls._process_single_photo(str(path), i, target_lang)
            except Exception as e:
                yield (i, f"Ошибка обработки: {str(e)}")


    @classmethod
    @abstractmethod
    def _process_single_photo(cls, photo_path: str, index: int, target_lang: str) -> Generator:
        pass


    @classmethod
    def safe_copy_file(src: Path, dest: Path) -> str:
        """Потокобезопасное копирование файла с проверкой"""
        try:
            if not src.exists():
                return f"Ошибка: файл не найден {src}"
                
            if dest.exists():
                return f"Файл уже существует: {dest}"
                
            shutil.copy(str(src), str(dest))
            return f"Успешно сохранён: {dest}"
            
        except Exception as e:
            return f"Ошибка копирования {src}: {e}"
        