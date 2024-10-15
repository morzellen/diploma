import torch
from modules.utils import logger
from transformers import (AutoProcessor, AutoTokenizer, AutoImageProcessor,
                          AutoModelForCausalLM, BlipForConditionalGeneration,
                          VisionEncoderDecoderModel, Qwen2VLForConditionalGeneration,
                          BertTokenizerFast)

class CaptioningModelCreator:
    def __init__(self, captioning_model_name, device):
        self.captioning_model_name = captioning_model_name
        self.device = device
        self.processor, self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Загрузка модели, процессора и токенизатора"""
        model_mapping = {
            "git-base-coco": ("microsoft/git-base-coco", AutoModelForCausalLM),
            "git-large-coco": ("microsoft/git-large-coco", AutoModelForCausalLM),
            "blip-image-captioning-base": ("Salesforce/blip-image-captioning-base", BlipForConditionalGeneration),
            "blip-image-captioning-large": ("Salesforce/blip-image-captioning-large", BlipForConditionalGeneration),
            "vit-gpt2-image-captioning": ("nlpconnect/vit-gpt2-image-captioning", VisionEncoderDecoderModel),
            "Qwen2-VL-7B-Instruct-GPTQ-Int4": ("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", Qwen2VLForConditionalGeneration)
        }

        if self.captioning_model_name not in model_mapping:
            raise ValueError(f"Модель {self.captioning_model_name} не поддерживается")

        model_path, model_class = model_mapping[self.captioning_model_name]

        try:
            # Проверяем, какая модель используется, и выбираем соответствующий процессор
            if model_class == VisionEncoderDecoderModel:
                processor = AutoImageProcessor.from_pretrained(model_path)
            else:
                processor = AutoProcessor.from_pretrained(model_path)
            
            model = model_class.from_pretrained(model_path).to(self.device)
            
            if model_class == BlipForConditionalGeneration:
                tokenizer = BertTokenizerFast.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"Модель {self.captioning_model_name} успешно загружена")
            return processor, model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    