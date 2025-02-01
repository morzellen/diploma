from core.utils.get_logger import logger
from core.constants.models import CAPTIONING_MODEL_NAMES
from transformers import (AutoProcessor, AutoTokenizer, AutoImageProcessor,
                          BlipForConditionalGeneration,
                          VisionEncoderDecoderModel,
                          BertTokenizerFast)

class CaptioningModelCreator:
    def __init__(self, captioning_model_name, device):
        self.captioning_model_name = captioning_model_name
        self.device = device
        self.processor, self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Загрузка модели, процессора и токенизатора"""
        if self.captioning_model_name not in CAPTIONING_MODEL_NAMES:
            raise ValueError(f"Модель {self.captioning_model_name} не поддерживается")

        model_path, model_class_name = CAPTIONING_MODEL_NAMES[self.captioning_model_name]
        model_class = globals()[model_class_name]

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
    