# src\core\creators\captioning_model_creator.py
import torch
from core.utils.get_logger import logger
from core.constants.models import CAPTIONING_MODEL_NAMES
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor, 
    AutoTokenizer, 
    AutoImageProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    BertTokenizerFast
)

class CaptioningModelCreator:
    _model_cache = {}

    def __init__(self, captioning_model_name, device):
        self.captioning_model_name = captioning_model_name
        self.device = device
        self.processor, self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        """Загрузка модели"""
        cache_key = (self.captioning_model_name, self.device)

        if cache_key not in self._model_cache:
            if self.captioning_model_name not in CAPTIONING_MODEL_NAMES:
                raise ValueError(f"Модель {self.captioning_model_name} не поддерживается")

            model_path, model_class_name = CAPTIONING_MODEL_NAMES[self.captioning_model_name]
            model_class = globals()[model_class_name]

            try:
                # Загрузка процессора
                if model_class == VisionEncoderDecoderModel:
                    processor = AutoImageProcessor.from_pretrained(model_path)
                else:
                    processor = AutoProcessor.from_pretrained(model_path)

                # Загрузка модели
                model = model_class.from_pretrained(model_path)
                model = model.to(self.device)

                # Оптимизации для GPU
                if self.device == 'cuda':
                    if model.dtype != torch.float16 and model.config.torch_dtype == torch.float16:
                        model = model.half()
                    if torch.__version__ >= "2.0.0":
                        try:
                            model = torch.compile(model)
                        except Exception as compile_error:
                            logger.warning(f"Ошибка компиляции модели: {compile_error}")

                # Загрузка токенизатора
                tokenizer = (
                    BertTokenizerFast.from_pretrained(model_path)
                    if model_class == BlipForConditionalGeneration
                    else AutoTokenizer.from_pretrained(model_path)
                )

                logger.info(f"Модель {self.captioning_model_name} загружена")
                self._model_cache[cache_key] = (processor, model, tokenizer)
            except Exception as e:
                logger.error(f"Ошибка загрузки: {e}")
                raise

        return self._model_cache[cache_key]
    

    