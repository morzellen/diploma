import torch
from modules.utils import logger
from transformers import (AutoProcessor, AutoTokenizer, AutoImageProcessor,
                          AutoModelForCausalLM, BlipForConditionalGeneration,
                          VisionEncoderDecoderModel, Qwen2VLForConditionalGeneration,
                          BertTokenizerFast)

class SegmentationModelCreator:
    def __init__(self, segmentation_model_name, device):
        self.segmentation_model_name = segmentation_model_name
        self.device = self.device
        self.processor, self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Загрузка модели, процессора и токенизатора"""
        model_mapping = {
            "segmentation_model_name": ("model_path", model_class),
        }

        if self.segmentation_model_name not in model_mapping:
            raise ValueError(f"Модель {self.segmentation_model_name} не поддерживается")

        model_path, model_class = model_mapping[self.segmentation_model_name]

        try:
            processor = None
            model = None
            tokenizer = None
            
            logger.info(f"Модель {self.segmentation_model_name} успешно загружена")
            return processor, model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    