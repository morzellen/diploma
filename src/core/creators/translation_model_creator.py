# src/core/creators/translation_model_creator.py
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoProcessor
)
from core.creators.base_creator import BaseCreator

class TranslationModelCreator(BaseCreator):
    _model_cache = {}

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.processor, self.tokenizer, self.model = self._load_model()

    @property
    def MODEL_NAMES(self):
        from core.constants.models import TRANSLATION_MODEL_NAMES
        return TRANSLATION_MODEL_NAMES

    def _load_components(self, model_path, model_class):
        # Загрузка процессора
        processor = (
            AutoProcessor.from_pretrained(model_path)
            if model_class != MBartForConditionalGeneration
            else None
        )
        
        # Загрузка токенизатора
        tokenizer = (
            MBart50TokenizerFast.from_pretrained(model_path)
            if model_class == MBartForConditionalGeneration
            else AutoTokenizer.from_pretrained(model_path)
        )
        
        # Загрузка модели
        model = self._load_base_model(model_path, model_class)
        
        return processor, tokenizer, model

    def _load_model(self):
        cache_key = (self.model_name, self.device)
        if cache_key not in self._model_cache:
            model_path, model_class = self._get_model_path_and_class()
            self._model_cache[cache_key] = self._load_components(model_path, model_class)
        return self._model_cache[cache_key]