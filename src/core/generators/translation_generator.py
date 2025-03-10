# src/core/generators/translation_generator.py
import contextlib
from functools import lru_cache
import torch
from core.creators.translation_model_creator import TranslationModelCreator
from core.generators.base_generator import BaseGenerator
from core.utils.get_logger import logger
from core.constants.web import TRANSLATION_LANGUAGES

class TranslationGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        """Инициализация генератора перевода с указанной моделью."""
        super().__init__()
        self.model = TranslationModelCreator(model_name, self.device)
        self.lang_cache = {}
        self._precache_language_ids()

    def _precache_language_ids(self):
        """Кэширование идентификаторов языков для ускорения работы."""
        for lang in TRANSLATION_LANGUAGES.values():
            self.lang_cache[lang] = self.model.tokenizer.lang_code_to_id[lang]

    @lru_cache(maxsize=500)
    def generate(self, text: str, src_lang: str, tgt_lang_str: str) -> str:
        """Перевод текста с исходного языка на целевой."""
        try:
            tgt_lang = TRANSLATION_LANGUAGES[tgt_lang_str]
            forced_bos_id = self.lang_cache[tgt_lang]
            inputs = self._prepare_inputs(text, src_lang)
            outputs = self._generate_translation(inputs, forced_bos_id)
            return self._decode_output(outputs)
        except Exception as e:
            logger.error(f"Ошибка перевода: {str(e)}")
            return text
        finally:
            self.handle_memory()

    def _prepare_inputs(self, text: str, src_lang: str):
        """Подготовка текста для модели."""
        self.model.tokenizer.src_lang = src_lang
        return self.model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

    def _generate_translation(self, inputs, forced_bos_id):
        """Генерация перевода с использованием модели."""
        with torch.inference_mode():
            # Автокаст только для CUDA
                with torch.autocast(device_type=self.device) if self.device == "cuda" else contextlib.nullcontext():
                    return self.model.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_id,
                        max_new_tokens=64,
                        num_beams=2,
                        early_stopping=True
                    )

    def _decode_output(self, generated_tokens):
        """Декодирование токенов в текст."""
        decoded = self.model.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        return decoded[0] if decoded else ""
    