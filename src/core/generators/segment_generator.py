# src\core\generators\segment_generator.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, List, Tuple

import torch
from core.utils.get_logger import logger
from core.creators.segmentation_model_creator import SegmentationModelCreator
from PIL import Image, UnidentifiedImageError

class SegmentGenerator():
    def __init__(self, segmentation_model_name, device):
        self.used_model = SegmentationModelCreator(segmentation_model_name, device)
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=4)

    @lru_cache(maxsize=100)
    def _load_image(self, photo_path: str) -> Image.Image:
        """Кэшированная загрузка и предобработка изображения"""
        try:
            image = Image.open(photo_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {photo_path}: {str(e)}")
            raise

    def _process_inputs(self, image: Image.Image):
        """Оптимизированная обработка входных данных"""
        return self.used_model.processor(
            text="<OD>",
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device, memory_format=torch.channels_last)

    def generate_segments(self, photo_path: str, photo_name: str) -> List[Tuple[str, List[float]]]:
        """Оптимизированная генерация сегментов с управлением памятью"""
        try:
            # Асинхронная загрузка изображения
            image = self._load_image(photo_path)
            
            # Пакетная обработка с использованием менеджера контекста
            with torch.inference_mode():
                inputs = self._process_inputs(image)
                outputs = self.used_model.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_beams=2,
                    early_stopping=True
                )

            # Постобработка результатов
            detection_text = self.used_model.processor.batch_decode(
                outputs, 
                skip_special_tokens=False
            )[0]
            
            parsed_answer = self.used_model.processor.post_process_generation(
                detection_text,
                task="<OD>",
                image_size=image.size
            )

            # Обработка результатов
            return self._parse_detections(parsed_answer)

        except Exception as e:
            logger.error(f"Ошибка обработки {photo_name}: {str(e)}")
            return []
        finally:
            # Очистка памяти CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _parse_detections(self, parsed_answer):
        """Векторизованная обработка результатов детекции"""
        detections = []
        if '<OD>' in parsed_answer:
            od_results = parsed_answer['<OD>']
            for bbox, label in zip(od_results['bboxes'], od_results['labels']):
                detections.append((label.lower(), bbox))
        return detections


    def get_main_object(self, detections: List[Tuple[str, List[float]]]) -> str:
        """Векторизованный расчет главного объекта"""
        if not detections:
            return "unknown"
            
        areas = [(self.get_box_area(coords), obj) for obj, coords in detections]
        return max(areas, key=lambda x: x[0])[1]

    @staticmethod
    def get_box_area(coords: List[float]) -> float:
        return (coords[2] - coords[0]) * (coords[3] - coords[1])
              

