# src\core\creators\segmentation_model_creator.py
import torch
from core.utils.get_logger import logger
from core.constants.models import SEGMENTATION_MODEL_NAMES
from transformers import AutoProcessor, AutoModelForCausalLM

class SegmentationModelCreator:
    _model_cache = {}
    

    def __init__(self, segmentation_model_name, device):
        self.segmentation_model_name = segmentation_model_name
        self.device = device
        self.processor, self.model = self._load_cached_model()

    def _load_cached_model(self):
        """Кэшированная загрузка модели с оптимизациями для GPU"""
        cache_key = (self.segmentation_model_name, self.device)
        
        if cache_key not in SegmentationModelCreator._model_cache:
            if self.segmentation_model_name not in SEGMENTATION_MODEL_NAMES:
                raise ValueError(f"Модель {self.segmentation_model_name} не поддерживается")

            model_path, model_class_name = SEGMENTATION_MODEL_NAMES[self.segmentation_model_name]
            model_class = globals()[model_class_name]
            
            try:
                # Загрузка процессора
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )

                # Конфигурация загрузки модели
                torch_dtype = torch.float16 if self.device == 'cuda' else None
                model = model_class.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype
                ).to(self.device)

                # Оптимизации для GPU
                if self.device == 'cuda':
                    model = model.to(memory_format=torch.channels_last)
                    if torch.__version__ >= "2.0.0":
                        model = torch.compile(model)

                logger.info(f"Модель {self.segmentation_model_name} загружена")
                SegmentationModelCreator._model_cache[cache_key] = (processor, model)
                
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise

        return SegmentationModelCreator._model_cache[cache_key]
    

