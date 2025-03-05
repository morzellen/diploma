# src/core/handlers/renaming_handler.py
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.generators.caption_generator import CaptionGenerator
from core.generators.translation_generator import TranslationGenerator
from core.handlers.base_handler import BaseHandler


class RenamingHandler(BaseHandler):
    _caption_generator = None
    _translation_generator = None

    @classmethod
    def initialize_models(cls, caption_model, trans_model):
        cls._caption_generator = CaptionGenerator(caption_model)
        cls._translation_generator = TranslationGenerator(trans_model)

    @classmethod
    def handle_photo_generator(cls, photo_tuple, caption_model, trans_model, check_cancelled, target_language):
        cls.initialize_models(caption_model, trans_model)
        yield from super()._common_processing(photo_tuple, check_cancelled, target_language)

    @classmethod
    def _generate_object(cls, photo_path, photo_name):
        return cls._caption_generator.generate(photo_path, photo_name)

    @classmethod
    def _translate_object(cls, generated_object, target_lang):
        return cls._translation_generator.generate(generated_object, "en_XX", target_lang)

    @classmethod
    def save_photo(cls, new_names, photo_paths, save_dir):
        save_path = Path(save_dir)
        if len(photo_paths) > 1:
            save_path /= 'renamed_photos'
        
        save_path.mkdir(parents=True, exist_ok=True)
        photo_paths = [Path(p) for p in photo_paths]
        name_counter = defaultdict(int)

        with ThreadPoolExecutor() as executor:
            futures = []
            for new_name, path in zip(new_names, photo_paths):
                base_name = new_name.strip()
                name_counter[base_name] += 1
                count = name_counter[base_name]
                
                final_name = f"{base_name} {count}" if count > 1 else base_name
                dest = save_path / f"{final_name}{path.suffix}"
                
                futures.append(executor.submit(
                    cls._safe_copy_file,
                    path,
                    dest
                ))
            
            return [future.result() for future in as_completed(futures)]