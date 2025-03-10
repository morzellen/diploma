# src/core/generators/segment_generator.py
from __future__ import annotations
import contextlib
from typing import List, Optional, Tuple
from transformers import BatchEncoding

import torch
from PIL import Image, UnidentifiedImageError

from core.creators.segmentation_model_creator import SegmentationModelCreator
from core.generators.base_generator import BaseGenerator
from core.utils.get_logger import logger
from core.generators.exceptions import ImageProcessingError, SegmenatationGenerationError


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
        logger.debug(f"Начало обработки {image_name}")

        try:
            image = self._process_image(image_path, image_name)
            inputs = self._prepare_inputs(image)
            outputs = self._generate_segments(inputs)
            detections = self._postprocess(outputs, image.size, image_name)
            result = self._get_main_object(detections)
            
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
        inputs = self.model_creator.processor(
            text="<OD>",
            images=image, 
            return_tensors="pt",
            padding=True
        )
        return inputs.to(self.device, non_blocking=True)
        
    def _generate_segments(self, inputs: BatchEncoding) -> torch.Tensor:
        """Генерация сегментов.
        
        Returns:
            torch.Tensor: Сгенерированные идентификаторы сегментов.
        """
        params = self.generation_params
        logger.debug(
            f"Начало генерации сегментов с параметрами: "
            f"max_new_tokens={params['max_new_tokens']}, num_beams={params['num_beams']}"
        )
        
        try:
            with torch.inference_mode():
            # Автокаст только для CUDA
                with torch.autocast(device_type=self.device) if self.device == "cuda" else contextlib.nullcontext():
                    return self.model_creator.model.generate(
                        **inputs,
                        max_new_tokens=params['max_new_tokens'],
                        num_beams=params['num_beams'],
                        early_stopping=True,
                        no_repeat_ngram_size=3,  # для уменьшения вычислений
                        length_penalty=0.8        # Ускорение генерации
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
        return [d for d in detections if d[1][2] > 0.1 and d[1][3] > 0.1]  # Фильтр мелких объектов

    def _get_main_object(self, detections: List[Tuple[str, List[float]]]) -> str:
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
    