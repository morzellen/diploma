from core.utils.get_logger import logger
from core.creators.translation_model_creator import TranslationModelCreator
from core.constants.web import TRANSLATION_LANGUAGES

class TranslationGenerator:
    def __init__(self, translating_model_name, device):
        self.used_model = TranslationModelCreator(translating_model_name, device)

    # def set_languages(self, src_lang, tgt_lang) -> None:
    #     """Изменение исходного и целевого языков перевода."""
    #     logger.info(f"Языки перевода изменены: исходный - {src_lang}, целевой - {tgt_lang}")

    def _unpack_lang_str(self, tgt_lang_str) -> str:
        return TRANSLATION_LANGUAGES[tgt_lang_str]

    def translate(self, original_name, src_lang, tgt_lang_str) -> str:
        """Перевод названия."""
        tgt_lang = self._unpack_lang_str(tgt_lang_str)
        # Выполняем перевод
        self.used_model.tokenizer.src_lang = src_lang
        encoded_lang = self.used_model.tokenizer(original_name, return_tensors="pt")
        generated_tokens = self.used_model.model.generate(
            **encoded_lang,
            forced_bos_token_id=self.used_model.tokenizer.lang_code_to_id[tgt_lang]
        )

        # Определяем, какой декодер использовать
        if self.used_model.tokenizer:
            translated = self.used_model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            translated = self.used_model.processor.batch_decode(generated_tokens, skip_special_tokens=True)

        translated = ''.join(translated)
        logger.info(f"Успешный перевод: '{original_name}' -> '{translated}'")
        return translated
