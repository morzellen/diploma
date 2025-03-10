# src\core\creators\segmentation_model_creator.py
import torch
from transformers import AutoProcessor
from core.creators.base_creator import BaseCreator

class SegmentationModelCreator(BaseCreator):
    _model_cache = {}
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.processor, self.model = self._load_model()

    @property
    def MODEL_NAMES(self):
        from core.constants.models import SEGMENTATION_MODEL_NAMES
        return SEGMENTATION_MODEL_NAMES

    def _load_components(self, model_path, model_class):
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = self._load_base_model(model_path, model_class, trust_remote_code=True)
        model = self._apply_special_optimizations(model)
        return processor, model

    def _apply_special_optimizations(self, model):
        if self.device == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        return model

    def _load_model(self):
        cache_key = (self.model_name, self.device)
        if cache_key not in self._model_cache:
            model_path, model_class = self._get_model_path_and_class()
            self._model_cache[cache_key] = self._load_components(model_path, model_class)
        return self._model_cache[cache_key]