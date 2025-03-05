# src/core/generators/segment_generator.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BatchEncoding

from core.creators.segmentation_model_creator import SegmentationModelCreator
from core.generators.base_generator import BaseGenerator
from core.utils.get_logger import logger
from core.exceptions import ImageProcessingError, SegmenatationGenerationError


class SegmentGenerator(BaseGenerator):
    """Генератор сегментации изображений с оптимизированной обработкой."""
    
    def __init__(self, model_name: str, max_new_tokens: int = 512, num_beams: int = 2):
        """Инициализирует генератор сегментации.
        
        Args:
            model_name: Название предобученной модели
        """
        super().__init__()
        self.model_creator = SegmentationModelCreator(model_name, self.device)

        self.generation_params = {
            'max_new_tokens': max_new_tokens,
            'num_beams': num_beams,
        }

    def generate(self, image_path: str, image_name: Optional[str] = None) -> List[Tuple[str, List[float]]]:
        """Основной метод генерации сегментов.
        
        Args:
            image_path: Путь к изображению
            image_name: Имя изображения (опционально)
        
        Returns:
            Список кортежей (метка, координаты bbox)
        
        Raises:
            SegmenatationGenerationError: При ошибках обработки
        """
        image_name = image_name or Path(image_path).name
        logger.debug(f"Начало обработки {image_name}")

        try:
            image = self._process_image(image_path, image_name)
            inputs = self._prepare_inputs(image)
            outputs = self._generate_segments(inputs)
            result = self._postprocess(outputs, image.size, image_name)
            
            logger.info(f"Успешная генерация для {image_name} | Результат: {result}")
            return result
            
        except ImageProcessingError as e:
            logger.error(f"Критическая ошибка обработки изображения {image_name}: {str(e)}")
            raise SegmenatationGenerationError(
                f"Ошибка обработки изображения для {image_name}"
            ) from e
        except Exception as e:
            logger.critical(f"Непредвиденная ошибка при обработке {image_name}: {str(e)}", exc_info=True)
            raise SegmenatationGenerationError(
                f"Ошибка генерации сегментов для {image_name}"
            ) from e
        finally:
            self.handle_memory()

    def _process_image(self, image_path: str, image_name: str) -> Image.Image:
        """Обработка изображения перед генерацией сегментов.
        
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
                # if img.mode not in ('RGB', 'L'):
                if img.mode != 'RGB':
                    logger.warning(
                        f"Конвертация изображения {image_name} из режима {img.mode} в RGB"
                    )
                    img = img.convert('RGB')
                
                logger.debug(
                    f"Изображение {image_name} загружено | "
                    f"Размер: {img.size} | Исходный режим: {img.mode}"
                )
                
                return img
            
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Некорректное изображение: {image_name}", exc_info=True)
            raise ImageProcessingError from e

    def _prepare_inputs(self, image: Image.Image) -> BatchEncoding:
        """Подготовка данных для модели."""
        return self.model_creator.processor(
            text="<OD>",
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device, non_blocking=True)
        
    def _generate_segments(self, inputs: BatchEncoding) -> torch.Tensor:
        """Генерация сегментов.
        
        Args:
            inputs (BatchEncoding): Обработанное изображение в виде тензора.
        
        Returns:
            torch.Tensor: Сгенерированные идентификаторы сегментов.
        """
        params = self.generation_params
        logger.debug(
            f"Начало генерации сегментов с параметрами: "
            f"max_new_tokens={params['max_new_tokens']}, num_beams={params['num_beams']}"
        )
        
        with torch.inference_mode(), torch.autocast(device_type=self.device):
            try:
                return self.model_creator.model.generate(
                    **inputs,
                    max_new_tokens=params['max_new_tokens'],
                    num_beams=params['num_beams'],
                    early_stopping=True,
                )
            except RuntimeError as e:
                logger.error(f"Ошибка генерации: {str(e)}", exc_info=True)
                raise SegmenatationGenerationError("Ошибка во время генерации сегментов") from e

    def _postprocess(self, outputs: torch.Tensor, image_size: Tuple[int, int], image_name: str) -> List[Tuple[str, List[float]]]:
        """Постобработка результатов."""
        detection_text = self.model_creator.processor.batch_decode(
            outputs, 
            skip_special_tokens=False
        )[0]
        
        parsed = self.model_creator.processor.post_process_generation(
            detection_text,
            task="<OD>",
            image_size=image_size
        )
        
        logger.debug(f"Результаты для {image_name}: {parsed.get('<OD>', {}).get('labels', [])}")
        return self._parse_detections(parsed)

    def _parse_detections(self, parsed_data: dict) -> List[Tuple[str, List[float]]]:
        """Парсинг результатов детекции."""
        detections = []
        if od_results := parsed_data.get('<OD>'):
            for label, bbox in zip(od_results['labels'], od_results['bboxes']):
                detections.append((label.lower(), [round(coord, 2) for coord in bbox]))
        return detections

    @staticmethod
    def get_main_object(detections: List[Tuple[str, List[float]]]) -> str:
        """Определение главного объекта по площади."""
        if not detections:
            return "unknown"
            
        max_area = -1
        main_obj = "unknown"
        for obj, coords in detections:
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            if area > max_area:
                max_area = area
                main_obj = obj
        return main_obj