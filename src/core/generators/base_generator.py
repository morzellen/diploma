# src\core\generators\base_generator.py
from typing import Literal, ClassVar
from abc import ABC, abstractmethod
from torch.cuda import empty_cache
from core.utils.get_device import get_device

class BaseGenerator(ABC):
    """Abstract base class for all generation components."""
    
    device: ClassVar[Literal["cuda", "cpu"]] = get_device()
    
    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """Main generation method to be implemented by subclasses"""
        pass
        
    @classmethod
    def handle_memory(cls) -> None:
        """Clean up GPU memory if available"""
        if cls.device == "cuda":
            empty_cache()
        else:
            import gc
            gc.collect()


