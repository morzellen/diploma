# src/core/handlers/classification_handler.py
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.generators.segment_generator import SegmentGenerator
from core.generators.translation_generator import TranslationGenerator
from core.handlers.base_handler import BaseHandler


class ClassificationHandler(BaseHandler):
    _segment_generator = None
    _translation_generator = None

    @classmethod
    def initialize_models(cls, seg_model, trans_model):
        cls._segment_generator = SegmentGenerator(seg_model)
        cls._translation_generator = TranslationGenerator(trans_model)

    @classmethod
    def handle_photo_generator(cls, photo_tuple, seg_model, trans_model, check_cancelled, target_language):
        cls.initialize_models(seg_model, trans_model)
        yield from super()._common_processing(photo_tuple, check_cancelled, target_language)

    @classmethod
    def _generate_object(cls, photo_path, photo_name):
        return cls._segment_generator.generate(photo_path, photo_name)

    @classmethod
    def _translate_object(cls, generated_object, target_lang):
        return cls._translation_generator.generate(generated_object, "en_XX", target_lang)

    @classmethod
    def save_photo(cls, class_names, photo_paths, save_dir):
        save_path = Path(save_dir) / 'classified_photos'
        save_path.mkdir(parents=True, exist_ok=True)
        photo_paths = [Path(p) for p in photo_paths]
        
        class_files = defaultdict(list)
        for class_name, photo_path in zip(class_names, photo_paths):
            class_files[class_name].append(photo_path)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for class_name, paths in class_files.items():
                class_dir = save_path / class_name
                class_dir.mkdir(exist_ok=True)
                
                for path in paths:
                    dest = class_dir / path.name
                    futures.append(executor.submit(
                        cls._safe_copy_file, 
                        path, 
                        dest
                    ))
            
            return [future.result() for future in as_completed(futures)]
