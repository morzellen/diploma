# src/core/creators/translation_model_creator.py
from transformers import (AutoTokenizer, MBartForConditionalGeneration, 
                          MBart50TokenizerFast, AutoProcessor)
from core.utils.get_logger import logger
from core.constants.models import TRANSLATION_MODEL_NAMES
import torch

class TranslationModelCreator:
    _model_cache = {}

    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.processor, self.tokenizer, self.model = self._load_cached_model()
    

    def _load_cached_model(self):
        """Кэшированная загрузка модели"""
        cache_key = (self.model_name, self.device)
        
        if cache_key not in TranslationModelCreator._model_cache:
            if self.model_name not in TRANSLATION_MODEL_NAMES:
                raise ValueError(f"Модель {self.model_name} не поддерживается")

            model_path, model_class_name = TRANSLATION_MODEL_NAMES[self.model_name]
            model_class = globals()[model_class_name]

            try:
                if model_class == MBartForConditionalGeneration:
                    processor = None
                else:
                    processor = AutoProcessor.from_pretrained(model_path)

                if model_class == MBartForConditionalGeneration:
                    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                model = model_class.from_pretrained(model_path)

                # Оптимизации для GPU
                model = model.to(self.device)
                if self.device == 'cuda':
                    model = model.half()
                    if torch.__version__ >= "2.0.0":
                        model = torch.compile(model)
                
                logger.info(f"Модель {self.model_name} загружена")
                TranslationModelCreator._model_cache[cache_key] = (processor, tokenizer, model)
                
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise

        return TranslationModelCreator._model_cache[cache_key]
    

