from utils.logger import logger
from googletrans import Translator

class NameTranslator:
    def __init__(self, src_lang, dest_lang):
        self.translator = Translator()
        self.src_lang = src_lang
        self.dest_lang = dest_lang

    def set_languages(self, src_lang, dest_lang) -> None:
        """Изменение исходного и целевого языков перевода."""
        self.src_lang = src_lang
        self.dest_lang = dest_lang
        logger.info(f"Языки перевода изменены: исходный - {src_lang}, целевой - {dest_lang}")

    def translate(self, original_name) -> str:
        """Перевод названия."""
        try:
            # Выполняем перевод
            translated = self.translator.translate(original_name, src=self.src_lang, dest=self.dest_lang).text

            logger.info(f"Успешный перевод: '{original_name}' -> '{translated}'")
            return translated
        except Exception as e:
            logger.error(f"Ошибка перевода '{original_name}': {e}")
            return original_name
