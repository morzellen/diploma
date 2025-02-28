from core.utils.get_logger import logger
from core.constants.models import TRANSLATION_MODEL_NAMES
from transformers import (AutoProcessor, AutoTokenizer,
                          MBartForConditionalGeneration, 
                          MBart50TokenizerFast)

class TranslationModelCreator:
    def __init__(self, translating_model_name, device):
        self.translating_model_name = translating_model_name
        self.device = device
        self.processor, self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Загрузка модели, процессора и токенизатора"""
        if self.translating_model_name not in TRANSLATION_MODEL_NAMES:
            raise ValueError(f"Модель {self.translating_model_name} не поддерживается")

        model_path, model_class_name = TRANSLATION_MODEL_NAMES[self.translating_model_name] 
        model_class = globals()[model_class_name]

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
