# src/core/generators/caption_generator.py
from __future__ import annotations
import contextlib
import re
from typing import ClassVar, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BatchEncoding

from core.creators.captioning_model_creator import CaptioningModelCreator
from core.generators.base_generator import BaseGenerator
from core.utils.get_logger import logger
from core.generators.exceptions import ImageProcessingError, CaptionGenerationError


class CaptionGenerator(BaseGenerator):
    """Генератор подписей к изображениям с оптимизированным процессом обработки."""
    
    UNWANTED_PATTERNS: ClassVar[Tuple[str, ...]] = (
        ".", "gif", "png", "jpg", "jpeg", "webp", "bmp", "tiff",
        "[ unused0 ]", "image of", "photo of", "picture of", "a screen shot",
        "a close up", "a drawing of", "a rendering of"
    )
    
    def __init__(self, model_name: str, max_length: int = 50, num_beams: int = 2):
        """Инициализирует генератор подписей к изображениям.

        Args:
            model_name (str): Название модели, которая будет использоваться для генерации подписей.
            max_length (int, optional): Максимальная длина генерируемой подписи. По умолчанию 50.
            num_beams (int, optional): Количество лучей (beams) для поиска в алгоритме beam search. 
                                    По умолчанию 2.

        Инициализирует:
            - Модель `CaptioningModelCreator`.
            - Регулярное выражение для фильтрации нежелательных паттернов в подписях.
            - Параметры генерации подписей (max_length и num_beams).
        """
        super().__init__()
        self.model_creator = CaptioningModelCreator(model_name, self.device)
        
        self._compiled_pattern = re.compile(
            r'(?:{})'.format('|'.join(map(re.escape, self.UNWANTED_PATTERNS))),
            flags=re.IGNORECASE
        )
        self.generation_params = {
            'max_length': max_length,
            'num_beams': num_beams,
        }

    def generate(self, image_path: str, image_name: Optional[str] = None) -> str:
        """Основной метод для генерации подписи к изображению.
        
        Args:
            image_path (str): Путь к изображению.
            image_name (Optional[str]): Имя изображения (если не указано, будет взято из пути).
        
        Returns:
            str: Сгенерированная подпись к изображению.
        
        Raises:
            CaptionGenerationError: Если произошла ошибка при обработке изображения или генерации подписи.
        """
        logger.debug(f"Начало обработки изображения: {image_name}")
        
        try:
            image = self._process_image(image_path, image_name)

            with torch.inference_mode():
            # Автокаст только для CUDA
                with torch.autocast(device_type=self.device) if self.device == "cuda" else contextlib.nullcontext():
                    inputs = self._prepare_inputs(image)
                    outputs = self._generate_caption(inputs)

            result = self._postprocess(outputs, image_name)
            
            logger.info(f"Успешная генерация для {image_name} | Результат: {result}")
            return result
            
        except ImageProcessingError as e:
            logger.error(f"Критическая ошибка обработки изображения {image_name}: {str(e)}")
            raise CaptionGenerationError(
                f"Ошибка обработки изображения для {image_name}"
            ) from e
        except Exception as e:
            logger.critical(f"Непредвиденная ошибка при обработке {image_name}: {str(e)}", exc_info=True)
            raise CaptionGenerationError(
                f"Ошибка генерации подписи для {image_name}"
            ) from e
        finally:
            self.handle_memory()

    def _process_image(self, image_path: str, image_name: str) -> Image.Image:
        """Загрузка и предобработка изображения.
        
        Args:
            image_path (str): Путь к изображению.
            image_name (str): Имя изображения (опционально).
        
        Returns:
            Image.Image: Объект изображения.
        
        Raises:
            ImageProcessingError: Если изображение не может быть обработано.
        """
        try:
            with Image.open(image_path) as img:

                if img.mode not in ('RGB', 'L'):
                    logger.warning(
                        f"Конвертация изображения {image_name} из режима {img.mode} в RGB"
                    )
                    img = img.convert('RGB')
                
                MAX_SIZE = 512  # Оптимальный размер для CPU
                img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)

                # возвращаем PIL.ImageFile.ImageFile вместо PIL.JpegImagePlugin.JpegImageFile
                img.load() # выделяет память для изображения и загружает его данные

                logger.debug(
                    f"Изображение {image_name} загружено | "
                    f"Размер: {img.size} | Исходный режим: {img.mode}"
                )

                return img
            
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Некорректное изображение: {image_name}", exc_info=True)
            raise ImageProcessingError(f"Ошибка загрузки изображения {image_name}: {str(e)}") from e

    def _prepare_inputs(self, image: Image.Image) -> BatchEncoding:
        """Подготовка данных для модели."""
        return self.model_creator.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device, non_blocking=True)

    def _generate_caption(self, inputs: BatchEncoding) -> torch.Tensor:
        """Генерация подписи к изображению с использованием модели.
        
        Args:
            inputs (BatchEncoding): Обработанное изображение в виде тензора.
        
        Returns:
            torch.Tensor: Сгенерированные идентификаторы подписи.
        """
        params = self.generation_params
        logger.debug(
            f"Начало генерации подписи с параметрами: "
            f"max_length={params['max_length']}, num_beams={params['num_beams']}"
        )

        try:
            return self.model_creator.model.generate(
                **inputs,
                max_length=params['max_length'],
                num_beams=params['num_beams'],
                early_stopping=True,
                no_repeat_ngram_size=3,  # для уменьшения вычислений
                length_penalty=0.8        # Ускорение генерации
            )
        except RuntimeError as e:
            logger.error(f"Ошибка генерации: {str(e)}", exc_info=True)
            raise CaptionGenerationError("Ошибка во время генерации подписи") from e

    def _postprocess(self, generated_ids: torch.Tensor, image_name: str) -> str:
        """Постобработка сгенерированной подписи.
        
        Args:
            generated_ids (torch.Tensor): Сгенерированные идентификаторы подписи.
            image_name (str): Имя изображения для логирования.
        
        Returns:
            str: Очищенная и отфильтрованная подпись.
        """
        caption = self.model_creator.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        filtered = self._compiled_pattern.sub('', caption)
        filtered = re.sub(r'\s+', ' ', filtered).strip()
        filtered = filtered.rstrip('.').strip()
        
        logger.debug(
            f"Постобработка для {image_name} | "
            f"Исходная подпись: '{caption}' | "
            f"Очищенная: '{filtered}'"
        )
        return filtered or "Не удалось сгенерировать подпись"
    