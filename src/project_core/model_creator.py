import torch
from utils.logger import logger
from transformers import (AutoProcessor, AutoTokenizer, AutoImageProcessor,
                          AutoModelForCausalLM, BlipForConditionalGeneration,
                          VisionEncoderDecoderModel)

class ModelCreator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = self._get_device()
        self.processor, self.model, self.tokenizer = self._load_model()

    def _get_device(self):
        """Определяем доступное устройство для модели"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используемое устройство: {device}")
        return device
    
    def _load_model(self):
        """Загрузка модели, процессора и токенизатора"""
        model_mapping = {
            "GIT-base": ("microsoft/git-base-coco", AutoModelForCausalLM),
            "GIT-large": ("microsoft/git-large-coco", AutoModelForCausalLM),
            "BLIP-base": ("Salesforce/blip-image-captioning-base", BlipForConditionalGeneration),
            "BLIP-large": ("Salesforce/blip-image-captioning-large", BlipForConditionalGeneration),
            "ViT+GPT-2": ("nlpconnect/vit-gpt2-image-captioning", VisionEncoderDecoderModel)
        }

        if self.model_name not in model_mapping:
            raise ValueError(f"    Модель {self.model_name} не поддерживается")

        model_path, model_class = model_mapping[self.model_name]

        try:
            # Проверяем, какая модель используется, и выбираем соответствующий процессор
            if model_class == VisionEncoderDecoderModel:
                processor = AutoImageProcessor.from_pretrained(model_path)
            else:
                processor = AutoProcessor.from_pretrained(model_path)
            
            model = model_class.from_pretrained(model_path).to(self.device)
            tokenizer = None
            if model_class in [AutoModelForCausalLM, VisionEncoderDecoderModel]:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"Модель {self.model_name} успешно загружена")
            return processor, model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    