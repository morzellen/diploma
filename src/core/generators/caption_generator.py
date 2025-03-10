# src/core/generators/caption_generator.py
from __future__ import annotations
import contextlib
import re
import time
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
        """Инициализирует генератор подписей к изображениям."""
        super().__init__()
        try:
            logger.info(
                f"Инициализация генератора | Модель: {model_name} "
                f"Параметры: max_length={max_length}, num_beams={num_beams}"
            )
            
            self.model_creator = CaptioningModelCreator(model_name, self.device)
            self._compiled_pattern = re.compile(
                r'(?:{})'.format('|'.join(map(re.escape, self.UNWANTED_PATTERNS))),
                flags=re.IGNORECASE
            )
            self.generation_params = {
                'max_length': max_length,
                'num_beams': num_beams,
            }
            
            logger.success("Генератор успешно инициализирован")
        except Exception as e:
            logger.critical(f"Ошибка инициализации генератора: {e}", exc_info=True)
            raise

    def generate(self, image_path: str, image_name: Optional[str] = None) -> str:
        """Основной метод для генерации подписи к изображению."""
        image_name = image_name or image_path.split("/")[-1]
        logger.info(f"Старт обработки: {image_name}")
        
        try:
            start_time = time.monotonic()
            
            image = self._process_image(image_path, image_name)
            logger.debug("Этап 1/4: Изображение обработано")
            
            inputs = self._prepare_inputs(image)
            logger.debug(f"Этап 2/4: Входные данные подготовлены | Ключи: {inputs.keys()}")
            
            outputs = self._generate_caption(inputs)
            logger.debug("Этап 3/4: Подпись сгенерирована")
            
            result = self._postprocess(outputs, image_name)
            logger.debug("Этап 4/4: Постобработка завершена")
            
            exec_time = time.monotonic() - start_time
            logger.success(
                f"Успешная генерация для '{image_name}' | "
                f"Результат: '{result}' | Время выполнения: {exec_time:.2f}с"
            )
            return result
            
        except ImageProcessingError as e:
            logger.error(f"Ошибка обработки изображения '{image_name}': {e}", exc_info=True)
            raise CaptionGenerationError(f"Сбой обработки изображения: {image_name}") from e
        except CaptionGenerationError as e:
            logger.error(f"Ошибка генерации подписи: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.critical(
                f"Критическая ошибка при обработке '{image_name}': {e}",
                exc_info=True
            )
            raise CaptionGenerationError("Непредвиденная ошибка генерации") from e
        finally:
            logger.debug("Очистка памяти")
            self.handle_memory()

    def _process_image(self, image_path: str, image_name: str) -> Image.Image:
        """Загрузка и предобработка изображения."""
        try:
            logger.info(f"Загрузка изображения: {image_name}")
            with Image.open(image_path) as img:
                # Контроль цветового режима
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
            logger.debug("Подготовка тензоров для модели")
            inputs = self.model_creator.processor(
                images=image,
                return_tensors="pt",
            ).to(self.device, non_blocking=True)
            
            logger.debug(f"Тензоры подготовлены | Устройство: {self.device}")
            return inputs
        except Exception as e:
            logger.error("Ошибка подготовки данных", exc_info=True)
            raise CaptionGenerationError("Сбой подготовки входных данных") from e

    def _generate_caption(self, inputs: BatchEncoding) -> torch.Tensor:
        """Генерация подписи к изображению."""
        params = self.generation_params
        logger.info(
            f"Запуск генерации | Параметры: "
            f"max_length={params['max_length']}, "
            f"num_beams={params['num_beams']}, "
            f"Устройство: {self.device}"
        )
        
        try:
            start_time = time.monotonic()
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
                        max_length=params['max_length'],
                        num_beams=params['num_beams'],
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=0.8
                    )
                    exec_time = time.monotonic() - start_time
                    logger.info(f"Генерация завершена | Время: {exec_time:.2f}с")
                    return outputs
        except RuntimeError as e:
            logger.error(f"Ошибка выполнения генерации: {e}", exc_info=True)
            raise CaptionGenerationError("Сбой в процессе генерации") from e
        except Exception as e:
            logger.error("Непредвиденная ошибка генерации", exc_info=True)
            raise CaptionGenerationError("Ошибка генерации") from e

    def _postprocess(self, generated_ids: torch.Tensor, image_name: str) -> str:
        """Постобработка сгенерированной подписи."""
        try:
            logger.debug("Декодирование результатов")
            caption = self.model_creator.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            logger.debug(f"Исходная подпись: '{caption}'")
            
            # Фильтрация нежелательных паттернов
            filtered = self._compiled_pattern.sub('', caption)
            filtered = re.sub(r'\s+', ' ', filtered).strip()
            filtered = filtered.rstrip('.').strip()
            
            # Проверка результата
            if not filtered:
                logger.warning("Пустая подпись после фильтрации | Исходный текст: '{caption}'")
                filtered = "Не удалось сгенерировать подпись"
            
            logger.info(
                f"Результат постобработки | Исходная: {len(caption)} симв. "
                f"Очищенная: {len(filtered)} симв."
            )
            return filtered
        except Exception as e:
            logger.error("Ошибка постобработки", exc_info=True)
            raise CaptionGenerationError("Сбой постобработки результатов") from e
        