# src\core\generators\base_generator.py
import gc
from typing import Literal, ClassVar
from abc import ABC, abstractmethod
import torch
from torch.cuda import empty_cache
from core.utils.get_device import get_device
from core.utils.get_logger import logger

class BaseGenerator(ABC):
    """Абстрактный базовый класс для всех компонентов генерации."""
    
    device: ClassVar[Literal["cuda", "cpu"]] = get_device()
    
    def __init__(self):
        logger.debug(f"Инициализация генератора на устройстве: {self.device}")

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """Основной метод генерации, должен быть реализован в подклассах"""
        pass
        
    @classmethod
    def handle_memory(cls) -> None:
        """Очистка памяти GPU при наличии"""
        try:
            if cls.device == "cuda":
                logger.info("Очистка видеопамяти GPU...")
                before = torch.cuda.memory_allocated()
                empty_cache()
                after = torch.cuda.memory_allocated()
                logger.success(
                    f"Память GPU освобождена. Использовалось: {before // 1024**2}MB -> "
                    f"Свободно: {after // 1024**2}MB"
                )
            else:
                logger.debug("Очистка оперативной памяти CPU...")
                before = gc.get_objects()
                gc.collect()
                after = gc.get_objects()
                logger.success(
                    f"Освобождено объектов памяти: {len(before) - len(after)}"
                )
        except Exception as e:
            logger.error(
                f"Ошибка при очистке памяти ({'GPU' if cls.device == 'cuda' else 'CPU'}): {e}",
                exc_info=True
            )
            raise
        