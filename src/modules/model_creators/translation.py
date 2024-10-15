import torch
from modules.utils import logger
from transformers import (AutoProcessor, AutoTokenizer,
                          AutoModelForCausalLM, BlipForConditionalGeneration,
                          VisionEncoderDecoderModel, Qwen2VLForConditionalGeneration,
                          BertTokenizerFast, MBartForConditionalGeneration, 
                          MBart50TokenizerFast)

class TranslationModelCreator:
    def __init__(self, translating_model_name, device):
        self.translating_model_name = translating_model_name
        self.device = device
        self.processor, self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Загрузка модели, процессора и токенизатора"""
        model_mapping = {
            "mbart-large-50-many-to-many-mmt": ("facebook/mbart-large-50-many-to-many-mmt", MBartForConditionalGeneration),

        }

        if self.translating_model_name not in model_mapping:
            raise ValueError(f"Модель {self.translating_model_name} не поддерживается")

        model_path, model_class = model_mapping[self.translating_model_name]

        try:
            # Проверяем, какая модель используется, и выбираем соответствующий процессор
            if model_class == MBartForConditionalGeneration:
                processor = None
            else:
                processor = AutoProcessor.from_pretrained(model_path)
            
            model = model_class.from_pretrained(model_path).to(self.device)

            if model_class == MBartForConditionalGeneration:
                tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info(f"Модель {self.translating_model_name} успешно загружена")
            return processor, model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
