# src\core\creators\base_creator.py
from abc import ABC, abstractmethod
import torch
from core.utils.get_logger import logger
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor, 
    AutoTokenizer, 
    AutoImageProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    BertTokenizerFast,
    MBartForConditionalGeneration,
    MBart50TokenizerFast
)

class BaseCreator(ABC):
    """
    Абстрактный базовый класс для создания генеративных моделей.
    Реализует общую логику загрузки моделей и компонентов.
    """
    
    @property
    @abstractmethod
    def MODEL_NAMES(self):
        pass

    @abstractmethod
    def _load_components(self, model_path, model_class):
        pass

    def _get_model_path_and_class(self):
        """Возвращает путь к модели и класс модели с обработкой ошибок"""
        try:
            if self.model_name not in self.MODEL_NAMES:
                error_msg = f"Модель {self.model_name} не поддерживается. Доступные модели: {list(self.MODEL_NAMES.keys())}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            model_path, model_class_name = self.MODEL_NAMES[self.model_name]
            logger.info(f"Выбрана модель: {self.model_name} | Путь: {model_path} | Класс: {model_class_name}")
            return model_path, globals()[model_class_name]
            
        except KeyError as ke:
            logger.exception(f"Ошибка получения класса модели: {ke}")
            raise
        except Exception as e:
            logger.critical(f"Неожиданная ошибка при выборе модели: {e}")
            raise

    def _load_base_model(self, model_path, model_class, **kwargs):
        """Базовый метод загрузки модели с улучшенным логированием"""
        logger.info(f"Начало загрузки модели {self.model_name} на устройство {self.device}")
        
        try:
            torch_dtype = torch.float32 if self.device == 'cpu' else torch.float16
            logger.debug(f"Установлен torch_dtype: {torch_dtype} для устройства {self.device}")

            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                **kwargs
            ).to(self.device)

            logger.success(f"Модель {self.model_name} успешно загружена на {self.device}")

            if self.device == 'cuda' and torch.__version__ >= "2.0.0":
                logger.info("Попытка компиляции модели с torch.compile()")
                try:
                    model = torch.compile(model)
                    logger.info("Модель успешно скомпилирована")
                except Exception as compile_error:
                    logger.warning(
                        f"Ошибка компиляции модели: {compile_error}. "
                        "Модель будет работать без компиляции. "
                        "Рекомендуется обновить версии torch/cuda при возможности"
                    )

            return model

        except IOError as ioe:
            logger.exception(f"Ошибка загрузки файлов модели: {ioe}")
            raise
        except RuntimeError as re:
            logger.exception(f"Ошибка выполнения при загрузке модели: {re}")
            raise
        except Exception as e:
            logger.critical(f"Критическая ошибка при загрузке модели: {e}")
            raise

    def _apply_special_optimizations(self, model):
        """Применение оптимизаций с базовым логированием"""
        logger.debug("Применение специальных оптимизаций к модели")
        try:
            # Реализация в подклассах может добавлять специфичные оптимизации
            return model
        except Exception as opt_error:
            logger.error(f"Ошибка при применении оптимизаций: {opt_error}")
            raise
