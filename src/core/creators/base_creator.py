from abc import ABC, abstractmethod
import torch
from core.utils.get_logger import logger
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor, 
    AutoTokenizer, 
    AutoImageProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    BertTokenizerFast,
    MBartForConditionalGeneration,
    MBart50TokenizerFast
)

class BaseCreator(ABC):
    """
    Абстрактный базовый класс для создания генеративных моделей.
    Реализует общую логику загрузки моделей и компонентов.
    """
    
    @property
    @abstractmethod
    def MODEL_NAMES(self):
        """Словарь поддерживаемых моделей вида {model_name: (model_path, model_class_name)}"""
        pass

    @abstractmethod
    def _load_components(self, model_path, model_class):
        """Абстрактный метод для загрузки специфичных компонентов модели"""
        pass

    def _get_model_path_and_class(self):
        """Возвращает путь к модели и класс модели"""
        if self.model_name not in self.MODEL_NAMES:
            raise ValueError(f"Модель {self.model_name} не поддерживается")
        
        model_path, model_class_name = self.MODEL_NAMES[self.model_name]
        return model_path, globals()[model_class_name]

    def _load_base_model(self, model_path, model_class, **kwargs):
        """Базовый метод загрузки модели с автоматическим выбором типа данных"""
        torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            **kwargs
        ).to(self.device)

        if self.device == 'cuda' and torch.__version__ >= "2.0.0":
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Ошибка компиляции модели: {e}")
        
        return model

    def _apply_special_optimizations(self, model):
        """Применение специфичных оптимизаций (может быть переопределено в подклассах)"""
        return model