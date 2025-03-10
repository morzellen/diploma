# src/core/generators/translation_generator.py
import contextlib
from functools import lru_cache
import torch
from core.creators.translation_model_creator import TranslationModelCreator
from core.generators.base_generator import BaseGenerator
from core.utils.get_logger import logger
from core.constants.web import TRANSLATION_LANGUAGES

from core.generators.exceptions import TranslationGenerationError

class TranslationGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        """Инициализация генератора перевода с указанной моделью."""
        super().__init__()
        try:
            logger.info(f"Инициализация переводчика | Модель: {model_name}")
            self.model = TranslationModelCreator(model_name, self.device)
            self.lang_cache = {}
            self._precache_language_ids()
            logger.success("Генератор перевода успешно инициализирован")
        except Exception as e:
            logger.critical(f"Ошибка инициализации переводчика: {e}", exc_info=True)
            raise

    def _precache_language_ids(self):
        """Кэширование идентификаторов языков для ускорения работы."""
        logger.debug("Предварительное кэширование языковых идентификаторов")
        try:
            for lang_code, lang_name in TRANSLATION_LANGUAGES.items():
                self.lang_cache[lang_name] = self.model.tokenizer.lang_code_to_id[lang_name]
                # logger.debug(f"Кэширован язык: {lang_name} -> ID: {self.lang_cache[lang_name]}")
            logger.info(f"Зарегистрировано языков: {len(self.lang_cache)}")
        except KeyError as ke:
            logger.error(f"Отсутствует языковой код в токенизаторе: {ke}", exc_info=True)
            raise TranslationGenerationError("Некорректная конфигурация языков") from ke

    @lru_cache(maxsize=500)
    def generate(self, text: str, src_lang: str, tgt_lang_str: str) -> str:
        """Перевод текста с исходного языка на целевой."""
        cache_key = (text, src_lang, tgt_lang_str)
        logger.info(
            f"Запрос перевода | Исходный язык: {src_lang} -> Целевой: {tgt_lang_str} | "
            f"Длина текста: {len(text)} символов"
        )
        
        try:
            if not text.strip():
                logger.warning("Получен пустой текст для перевода")
                return ""

            # Получение целевого языка
            if tgt_lang_str not in TRANSLATION_LANGUAGES:
                error_msg = f"Неподдерживаемый целевой язык: {tgt_lang_str}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            tgt_lang = TRANSLATION_LANGUAGES[tgt_lang_str]
            forced_bos_id = self.lang_cache.get(tgt_lang)
            
            if not forced_bos_id:
                error_msg = f"ID языка '{tgt_lang}' не найден в кэше"
                logger.error(error_msg)
                raise TranslationGenerationError(error_msg)

            # Подготовка и генерация
            inputs = self._prepare_inputs(text, src_lang)
            outputs = self._generate_translation(inputs, forced_bos_id)
            result = self._decode_output(outputs)
            
            logger.success(
                f"Успешный перевод | Символы: {len(text)}->{len(result)} | "
                f"Языки: {src_lang}->{tgt_lang_str}"
            )
            return result
            
        except Exception as e:
            logger.error(
                f"Ошибка перевода для текста '{text[:30]}...': {e}",
                exc_info=True
            )
            raise TranslationGenerationError("Ошибка выполнения перевода") from e
        finally:
            self.handle_memory()
            # Логирование кэша
            if self.generate.cache_info().currsize > 0 and self.generate.cache_info().hits > 0:
                logger.debug(
                    f"Статистика кэша: Попадания={self.generate.cache_info().hits} "
                    f"Промахи={self.generate.cache_info().misses}"
                )

    def _prepare_inputs(self, text: str, src_lang: str):
        """Подготовка текста для модели."""
        logger.debug("Токенизация входного текста")
        try:
            self.model.tokenizer.src_lang = src_lang
            inputs = self.model.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            logger.debug(
                f"Токены подготовлены | Размер: {inputs.input_ids.shape} | "
                f"Устройство: {self.model.device}"
            )
            return inputs
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}", exc_info=True)
            raise TranslationGenerationError("Ошибка обработки текста") from e

    def _generate_translation(self, inputs, forced_bos_id):
        """Генерация перевода с использованием модели."""
        logger.info(
            "Генерация перевода | "
            f"Параметры: max_new_tokens=64, num_beams=2, bos_token_id={forced_bos_id}"
        )
        try:
            with torch.inference_mode():
                context = (
                    torch.autocast(device_type=self.device) 
                    if self.device == "cuda" 
                    else contextlib.nullcontext()
                )
                with context:
                    logger.debug(f"Контекст генерации: {type(context).__name__}")
                    outputs = self.model.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_id,
                        max_new_tokens=64,
                        num_beams=2,
                        early_stopping=True
                    )
                    logger.debug(f"Сгенерированные токены: {outputs.shape}")
                    return outputs
        except RuntimeError as re:
            logger.error(f"Ошибка выполнения: {re}", exc_info=True)
            raise TranslationGenerationError("Ошибка генерации перевода") from re
        except Exception as e:
            logger.error(f"Непредвиденная ошибка генерации: {e}", exc_info=True)
            raise

    def _decode_output(self, generated_tokens):
        """Декодирование токенов в текст."""
        logger.debug("Декодирование выходных токенов")
        try:
            decoded = self.model.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
            result = decoded[0] if decoded else ""
            
            if not result:
                logger.warning("Пустой результат декодирования")
                
            logger.debug(f"Результат декодирования: {result[:60]}...")
            return result
        except Exception as e:
            logger.error(f"Ошибка декодирования: {e}", exc_info=True)
            raise TranslationGenerationError("Ошибка обработки результата") from e
        