from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    AutoImageProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    BertTokenizerFast
)
from core.creators.base_creator import BaseCreator

class CaptioningModelCreator(BaseCreator):
    _model_cache = {}
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.processor, self.model, self.tokenizer = self._load_model()

    @property
    def MODEL_NAMES(self):
        from core.constants.models import CAPTIONING_MODEL_NAMES
        return CAPTIONING_MODEL_NAMES

    def _load_components(self, model_path, model_class):
        # Загрузка процессора
        processor = (
            AutoImageProcessor.from_pretrained(model_path)
            if model_class == VisionEncoderDecoderModel
            else AutoProcessor.from_pretrained(model_path)
        )
        
        # Загрузка модели
        model = self._load_base_model(model_path, model_class)
        
        # Загрузка токенизатора
        tokenizer = (
            BertTokenizerFast.from_pretrained(model_path)
            if model_class == BlipForConditionalGeneration
            else AutoTokenizer.from_pretrained(model_path)
        )
        return processor, model, tokenizer

    def _load_model(self):
        cache_key = (self.model_name, self.device)
        if cache_key not in self._model_cache:
            model_path, model_class = self._get_model_path_and_class()
            self._model_cache[cache_key] = self._load_components(model_path, model_class)
        return self._model_cache[cache_key]
