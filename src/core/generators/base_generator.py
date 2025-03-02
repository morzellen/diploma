# src\core\generators\base_generator.py
from typing import Literal
import torch
from abc import ABC, abstractmethod
from torch.cuda import empty_cache as cuda_empty_cache
from core.utils.get_device import get_device

class BaseGenerator(ABC):
    """Abstract base class for all generation components."""
    
    def __init__(self, device: Literal["cpu", "cuda"] = get_device) -> None:
        self.device = device
        
    @abstractmethod
    def generate(self, *args, **kwargs):
        """Main generation method to be implemented by subclasses"""
        pass
        
    @staticmethod
    def handle_memory() -> None:
        """Clean up GPU memory if available"""
        if torch.cuda.is_available():
            cuda_empty_cache()