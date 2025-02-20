from core.utils.get_logger import logger
from core.constants.models import SEGMENTATION_MODEL_NAMES
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM
)

class SegmentationModelCreator:
    def __init__(self, segmentation_model_name, device):
        self.segmentation_model_name = segmentation_model_name
        self.device = device
        self.processor, self.model = self._load_model()
    
    def _load_model(self):
        """Загрузка модели и процессора"""

        if self.segmentation_model_name not in SEGMENTATION_MODEL_NAMES:
            raise ValueError(f"Модель {self.segmentation_model_name} не поддерживается")

        model_path, model_class_name = SEGMENTATION_MODEL_NAMES[self.segmentation_model_name]
        model_class = globals()[model_class_name]

        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            model = model_class.from_pretrained(
                model_path,
                trust_remote_code=True
            ).to(self.device)
            
            logger.info(f"Модель {self.segmentation_model_name} успешно загружена")
            return processor, model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    