import re
from typing import Dict, List, Tuple
from core.utils.get_logger import logger
from core.creators.segmentation_model_creator import SegmentationModelCreator
from PIL import Image, UnidentifiedImageError

class SegmentGenerator():
    def __init__(self, segmentation_model_name, device):
        self.used_model = SegmentationModelCreator(segmentation_model_name, device)
        self.device = device

    def generate_segments(self, photo_path: str, photo_name: str) -> List[Tuple[str, List[float]]]:
        """Генерация сегментов изображения"""
        try:
            # Обрабатываем изображение
            inputs = self.process_photo(photo_path, photo_name)
            
            # Генерируем сегменты с помощью специального токена <OD>
            outputs = self.used_model.model.generate(
                **inputs,
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

            # Декодируем результаты
            detection_text = self.used_model.processor.batch_decode(outputs, skip_special_tokens=False)[0]
            
            # Обрабатываем результаты
            image = Image.open(photo_path)
            parsed_answer = self.used_model.processor.post_process_generation(
                detection_text,
                task="<OD>",
                image_size=(image.width, image.height)
            )
            
            # Преобразуем результаты в нужный формат
            detections = []
            if '<OD>' in parsed_answer:
                od_results = parsed_answer['<OD>']
                for bbox, label in zip(od_results['bboxes'], od_results['labels']):
                    detections.append((label.lower(), bbox))
            
            logger.info(f"Обнаружено {len(detections)} сегментов в {photo_name}")
            return detections

        except Exception as e:
            logger.error(f"Ошибка при генерации сегментов для {photo_name}: {str(e)}")
            return []

    def process_photo(self, photo_path: str, photo_name: str):
        """Обработка изображения"""
        try:
            image = Image.open(photo_path)

            # Конвертация изображения в RGB формат, если необходимо
            if image.mode != 'RGB':
                logger.info(f"{photo_name} конвертирован в RGB")
                image = image.convert('RGB')
            
            # Обрабатываем изображение с помощью процессора
            inputs = self.used_model.processor(
                text="<OD>",  # специальный токен для детекции объектов
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            logger.info(f"{photo_name} успешно обработан")
            return inputs

        except UnidentifiedImageError:
            logger.error(f"Невозможно открыть изображение {photo_name}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при обработке {photo_name}: {str(e)}")
            raise

    def get_box_area(self, coords: List[float]) -> float:
        """Вычисляет площадь ограничивающей рамки"""
        width = coords[2] - coords[0]
        height = coords[3] - coords[1]
        return width * height

    def get_main_object(self, detections: List[Tuple[str, List[float]]]) -> str:
        """Определяет главный объект на изображении по площади сегмента"""
        if not detections:
            return "unknown"
            
        main_object = "unknown"
        max_area = 0
        
        for obj_class, coords in detections:
            area = self.get_box_area(coords)
            if area > max_area:
                max_area = area
                main_object = obj_class
        
        return main_object
        