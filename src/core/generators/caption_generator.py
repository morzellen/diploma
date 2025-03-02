# src\core\generators\caption_generator.py
# import re
# from core.utils.get_logger import logger
# from core.creators.captioning_model_creator import CaptioningModelCreator
# from PIL import Image, UnidentifiedImageError

# class CaptionGenerator:
#     UNWANTED_WORDS = [".", "gif", "png", "jpg", "jpeg", "[ unused0 ]"]
#     UNWANTED_PATTERN = re.compile(
#         r'\b(?:' + '|'.join(re.escape(word) for word in UNWANTED_WORDS) + r')\b', 
#         flags=re.IGNORECASE
#     )


#     def __init__(self, captioning_model_name, device):
#         self.creator = CaptioningModelCreator(captioning_model_name, device)
#         self.device = device


#     def generate_caption(self, photo_path, photo_name):
#         """Генерация описания с оптимизациями"""
#         try:
#             inputs = self._process_photo(photo_path, photo_name)
#             generated_ids = self.creator.generate(
#                 inputs.pixel_values, 
#                 max_length=50,
#                 num_beams=5  # Оптимизация поиска
#             )
#             caption = self.creator.decode(generated_ids)
#             return self._filter_caption(caption, photo_name)
#         except Exception as e:
#             logger.error(f"Ошибка генерации для {photo_name}: {e}")
#             return ""


#     def _process_photo(self, photo_path, photo_name):
#         """Оптимизированная обработка изображений"""
#         try:
#             with Image.open(photo_path) as img:
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
#                 inputs = self.creator.processor(
#                     images=img, 
#                     return_tensors="pt"
#                 ).to(self.device)
#                 logger.debug(f"Обработано: {photo_name}")
#                 return inputs
#         except Exception as e:
#             logger.error(f"Ошибка обработки {photo_name}: {e}")
#             raise


#     def _filter_caption(self, caption, photo_name):
#         """Фильтрация результата"""
#         filtered = self.UNWANTED_PATTERN.sub('', caption)
#         logger.info(f"Сгенерировано для {photo_name}: {filtered}")
#         return filtered.strip()
    

# src\core\generators\caption_generator.py
from __future__ import annotations
from pathlib import Path
import re
from typing import Literal, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BatchEncoding, PreTrainedTokenizerBase

from core.creators.captioning_model_creator import CaptioningModelCreator
from core.generators.base_generator import BaseGenerator
from core.utils.get_logger import logger
from core.exceptions import ImageProcessingError, CaptionGenerationError


class CaptionFilter:
    """Text processing component for caption post-processing"""
    
    def __init__(self, unwanted_patterns: Tuple[str, ...]):
        self._compiled_pattern = self._build_regex(unwanted_patterns)
    
    @staticmethod
    def _build_regex(patterns: Tuple[str, ...]) -> re.Pattern:
        return re.compile(
            r'\b(?:' + '|'.join(map(re.escape, patterns)) + r')\b', 
            flags=re.IGNORECASE
        )
    
    def apply(self, text: str) -> str:
        """Apply filtering to generated caption"""
        return self._compiled_pattern.sub('', text).strip()

class CaptionGenerator(BaseGenerator):
    """Advanced image captioning generator with optimized processing pipeline
    
    Attributes:
        processor (PreTrainedProcessor): Image/text processor
        model (PreTrainedModel): Loaded captioning model
        filter (CaptionFilter): Text post-processing component
        
    Example:
        >>> generator = CaptionGenerator("Salesforce/blip2-opt-2.7b", torch.device("cuda"))
        >>> caption = generator.generate_caption("path/to/image.jpg", "example_image")
    """
    
    UNWANTED_PATTERNS = (".", "gif", "png", "jpg", "jpeg", "[ unused0 ]")
    
    def __init__(
        self, 
        model_name: str,
        max_length: int = 50,
        num_beams: int = 2
    ):
        self.model_creator = CaptioningModelCreator(model_name, self.device)
        self.filter = CaptionFilter(self.UNWANTED_PATTERNS)
        self.generation_params = {
            'max_length': max_length,
            'num_beams': num_beams,
            'early_stopping': True
        }
        self._setup_components()

    def _setup_components(self) -> None:
        """Настройка компонентов модели с автоматической привязкой токенов"""
        with torch.no_grad():
            self.processor = self.model_creator.processor
            self.model = self.model_creator.model
            
            # Автоматическая настройка отсутствующих токенов
            tokenizer = self.processor.tokenizer
            config = self.model.config
            
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            if not tokenizer.sep_token:
                tokenizer.sep_token = tokenizer.eos_token
                
            self.model.eval()

    def generate(
        self,
        image_path: str,
        image_name: Optional[str] = None
    ) -> str:
        """End-to-end caption generation pipeline
        
        Args:
            image_path: Path to input image file
            image_name: Optional identifier for logging
            
        Returns:
            str: Processed caption text
            
        Raises:
            ImageProcessingError: If image loading fails
            CaptionGenerationError: If caption generation fails
        """
        image_name = image_name or Path(image_path).name
        
        try:
            inputs = self._process_image(image_path, image_name)
            outputs = self._generate_caption(inputs)
            return self._postprocess(outputs, image_name)
        except ImageProcessingError as e:
            raise CaptionGenerationError(
                f"Image processing failed for {image_name}"
            ) from e
        finally:
            self.handle_memory()

    def _process_image(
        self, 
        image_path: str,
        image_name: str
    ) -> BatchEncoding:
        """Optimized image preprocessing pipeline"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                return self.processor(
                    images=img,
                    return_tensors="pt"
                ).to(self.device, memory_format=torch.channels_last)
                
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Invalid image format: {image_path}")
            raise ImageProcessingError from e
            
        except Exception as e:
            logger.error(f"Unexpected error processing {image_name}: {str(e)}")
            raise ImageProcessingError from e

    def _generate_caption(self, inputs: BatchEncoding) -> torch.Tensor:
        """Безопасная генерация без дублирования параметров"""
        with torch.inference_mode(), torch.autocast(self.device):
            return self.model.generate(
                **inputs,
                **self.generation_params
            )

    def _postprocess(
        self, 
        generated_ids: torch.Tensor,
        image_name: str
    ) -> str:
        """Post-processing pipeline for generated output"""
        try:
            caption = self.processor.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            filtered = self.filter.apply(caption)
            
            logger.info(
                "Caption generated",
                extra={"image": image_name, "caption": filtered}
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"Post-processing failed for {image_name}: {str(e)}")
            return ""

