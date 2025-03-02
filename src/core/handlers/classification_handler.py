# src\core\handlers\classification_handler.py
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


from core.generators.segment_generator import SegmentGenerator
from core.generators.translation_generator import TranslationGenerator
from core.handlers.base_handler import BaseHandler
from core.utils.get_logger import logger


class ClassificationHandler(BaseHandler):
    _segmentor = None
    _translator = None

    @classmethod
    def initialize_models(cls, seg_model, trans_model):
        cls._segmentor = SegmentGenerator(seg_model)
        cls._translator = TranslationGenerator(trans_model)


    @classmethod
    def handle_photo_generator(cls, photo_tuple, seg_model, trans_model, check_cancelled, target_language):
        if not cls._segmentor or not cls._translator:
            cls.initialize_models(seg_model, trans_model)
        yield from super()._common_processing(photo_tuple, check_cancelled, target_language)


    @classmethod
    def _process_single_photo(cls, photo_path, index, target_language):
        photo_name = Path(photo_path).name
        yield f"Обработка: {photo_name}"
        
        # Оптимизированный вызов с освобождением памяти
        with torch.no_grad():
            detections = cls._segmentor.generate_segments(photo_path, photo_name)
            main_object = cls._segmentor.get_main_object(detections)
            translated = cls._translator.translate(main_object, "en_XX", target_language)
        
        # Принудительная очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        yield (index, (main_object, translated))


    @staticmethod
    def save_photo(class_names, photo_paths, save_dir):
        """Сохранение классифицированных изображений с оптимизацией IO операций"""
        save_path = Path(save_dir) / 'classified_photos'
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Преобразование путей в объекты Path
        photo_paths = [Path(p) for p in photo_paths]
        
        # Группировка файлов по классам
        class_files = defaultdict(list)
        for class_name, photo_path in zip(class_names, photo_paths):
            class_files[class_name].append(photo_path)
        
        # Параллельное копирование файлов
        with ThreadPoolExecutor() as executor:
            futures = []
            for class_name, paths in class_files.items():
                class_dir = save_path / class_name
                class_dir.mkdir(exist_ok=True)
                
                for path in paths:
                    dest = class_dir / path.name
                    futures.append(executor.submit(
                        super()._safe_copy_file, 
                        path, 
                        dest
                    ))
            
            return [future.result() for future in as_completed(futures)]
    

