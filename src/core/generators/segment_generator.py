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
from core.generators.exceptions import ImageProcessingError, SegmentationGenerationError


class SegmentGenerator(BaseGenerator):
    """Генератор сегментации изображений с оптимизированной обработкой."""
    
    def __init__(self, model_name: str, max_new_tokens: int = 512, num_beams: int = 2):
        """Инициализирует генератор сегментации."""
        super().__init__()
        try:
            logger.info(
                f"Инициализация генератора сегментации | Модель: {model_name} "
                f"Параметры: max_new_tokens={max_new_tokens}, num_beams={num_beams}"
            )
            self.model_creator = SegmentationModelCreator(model_name, self.device)
            self.generation_params = {
                'max_new_tokens': max_new_tokens,
                'num_beams': num_beams,
            }
            logger.success("Генератор успешно инициализирован")
        except Exception as e:
            logger.critical(f"Ошибка инициализации генератора: {e}", exc_info=True)
            raise

    def generate(self, image_path: str, image_name: Optional[str] = None) -> List[Tuple[str, List[float]]]:
        """Основной метод генерации сегментов."""
        logger.debug(f"Старт обработки изображения: {image_name or 'без имени'}")
        try:
            image = self._process_image(image_path, image_name)
            logger.debug("Этап 1/4: Изображение успешно обработано")
            
            inputs = self._prepare_inputs(image)
            logger.debug(f"Этап 2/4: Данные подготовлены | Формат: {inputs.keys()}")
            
            outputs = self._generate_segments(inputs)
            logger.debug("Этап 3/4: Сегменты сгенерированы")
            
            detections = self._postprocess(outputs, image.size, image_name)
            logger.debug(f"Этап 4/4: Постобработка завершена | Найдено объектов: {len(detections)}")
            
            result = self._get_main_object(detections)
            logger.success(
                f"Успешная генерация для '{image_name}' | "
                f"Главный объект: {result[0]} | Размер bbox: {result[1]:.2f}"
            )
            return result[0]
            
        except ImageProcessingError as e:
            logger.error(f"Ошибка обработки изображения '{image_name}': {e}", exc_info=True)
            raise SegmentationGenerationError(f"Сбой обработки изображения: {image_name}") from e
        except SegmentationGenerationError as e:
            logger.error(f"Ошибка генерации сегментов: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(
                f"Критическая ошибка при обработке '{image_name}': {e}",
                exc_info=True
            )
            raise SegmentationGenerationError("Непредвиденная ошибка генерации") from e
        finally:
            logger.debug("Запуск очистки памяти")
            self.handle_memory()

    def _process_image(self, image_path: str, image_name: str) -> Image.Image:
        """Обработка изображения перед генерацией сегментов."""
        try:
            logger.info(f"Загрузка изображения: {image_name}")
            with Image.open(image_path) as img:
                # Контроль режима изображения
                if img.mode != 'RGB':
                    logger.warning(f"Конвертация {img.mode} в RGB | Изображение: {image_name}")
                    img = img.convert('RGB')

                # Оптимизация размера
                MAX_SIZE = 512
                if any(dim > MAX_SIZE for dim in img.size):
                    logger.debug(f"Уменьшение размера {img.size} -> {MAX_SIZE}px | Алгоритм: LANCZOS")
                    img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)

                # Предзагрузка данных
                img.load()
                logger.info(
                    f"Изображение готово | Размер: {img.size} | "
                    f"Объем данных: {len(img.tobytes()) // 1024} KB"
                )
                return img
            
        except (UnidentifiedImageError, OSError) as e:
            logger.error(f"Некорректный файл изображения: {image_name}", exc_info=True)
            raise ImageProcessingError(f"Файл поврежден или не является изображением: {image_name}") from e
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {image_name}", exc_info=True)
            raise ImageProcessingError(f"Ошибка обработки: {image_name}") from e

    def _prepare_inputs(self, image: Image.Image) -> BatchEncoding:
        """Подготовка данных для модели."""
        try:
            logger.debug("Подготовка входных данных для модели")
            inputs = self.model_creator.processor(
                text="<OD>",
                images=image, 
                return_tensors="pt",
                padding=True
            )
            logger.debug(f"Тензоры подготовлены | Устройство: {self.device}")
            return inputs.to(self.device, non_blocking=True)
        except Exception as e:
            logger.error("Ошибка подготовки данных", exc_info=True)
            raise SegmentationGenerationError("Сбой подготовки входных данных") from e

    def _generate_segments(self, inputs: BatchEncoding) -> torch.Tensor:
        """Генерация сегментов."""
        params = self.generation_params
        logger.info(
            f"Запуск генерации | Параметры: "
            f"max_new_tokens={params['max_new_tokens']}, "
            f"num_beams={params['num_beams']}, "
            f"Устройство: {self.device}"
        )
        
        try:
            with torch.inference_mode():
                context = (
                    torch.autocast(device_type=self.device) 
                    if self.device == "cuda" 
                    else contextlib.nullcontext()
                )
                with context:
                    logger.debug("Контекст генерации активирован" + (" (autocast)" if self.device == "cuda" else ""))
                    outputs = self.model_creator.model.generate(
                        **inputs,
                        max_new_tokens=params['max_new_tokens'],
                        num_beams=params['num_beams'],
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=0.8
                    )
                    logger.debug("Генерация завершена успешно")
                    return outputs
        except RuntimeError as e:
            logger.error(f"Ошибка выполнения генерации: {e}", exc_info=True)
            raise SegmentationGenerationError("Сбой в процессе генерации") from e
        except Exception as e:
            logger.error("Непредвиденная ошибка генерации", exc_info=True)
            raise SegmentationGenerationError("Ошибка генерации") from e

    def _postprocess(self, outputs: torch.Tensor, image_size: Tuple[int, int], image_name: str) -> List[Tuple[str, List[float]]]:
        """Постобработка результатов."""
        try:
            logger.debug("Декодирование результатов модели")
            detection_text = self.model_creator.processor.batch_decode(
                outputs, 
                skip_special_tokens=False
            )[0]
            
            logger.debug("Постобработка данных")
            parsed = self.model_creator.processor.post_process_generation(
                detection_text,
                task="<OD>",
                image_size=image_size
            )
            
            logger.info(f"Сырые результаты: {parsed.get('<OD>', {}).get('labels', [])}")
            return self._parse_detections(parsed)
        except Exception as e:
            logger.error("Ошибка постобработки", exc_info=True)
            raise SegmentationGenerationError("Сбой постобработки результатов") from e

    def _parse_detections(self, parsed_data: dict) -> List[Tuple[str, List[float]]]:
        """Парсинг результатов детекции."""
        logger.debug("Анализ обнаруженных объектов")
        try:
            detections = []
            if od_results := parsed_data.get('<OD>'):
                for label, bbox in zip(od_results['labels'], od_results['bboxes']):
                    cleaned_label = label.lower().strip()
                    detections.append((cleaned_label, [round(coord, 2) for coord in bbox]))
            logger.debug(f"Найдено объектов до фильтрации: {len(detections)}")
            return [d for d in detections if d[1][2] > 0.1 and d[1][3] > 0.1]
        except KeyError as e:
            logger.error(f"Ошибка парсинга данных: отсутствует ключ {e}", exc_info=True)
            raise SegmentationGenerationError("Некорректный формат выходных данных") from e

    def _get_main_object(self, detections: List[Tuple[str, List[float]]]) -> Tuple[str, float]:
        """Определение главного объекта по площади."""
        logger.debug("Выбор главного объекта")
        if not detections:
            logger.warning("Не обнаружено объектов на изображении")
            return ("unknown", [0.0]*4)
            
        max_area = -1
        main_obj = "unknown"
        for obj, coords in detections:
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            if area > max_area:
                max_area = area
                main_obj = obj
        logger.info(f"Выбран главный объект: {main_obj} | Площадь: {max_area:.2f}")
        return main_obj, max_area
    