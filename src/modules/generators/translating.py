from modules.utils import logger
from modules.model_creators.translation import TranslationModelCreator

class TranslationGenerator:
    def __init__(self, translating_model_name, device):
        self.used_model = TranslationModelCreator(translating_model_name, device)

    # def set_languages(self, src_lang, tgt_lang) -> None:
    #     """Изменение исходного и целевого языков перевода."""
    #     logger.info(f"Языки перевода изменены: исходный - {src_lang}, целевой - {tgt_lang}")

    def _unpack_tgt_lang_str(self, tgt_lang_str) -> str:
        language_mapping = {
        "Afrikaans": "af_ZA", "Arabic": "ar_AR",
        "Azerbaijani": "az_AZ", "Bengali": "bn_IN",
        "Burmese": "my_MM", "Chinese": "zh_CN",
        "Croatian": "hr_HR", "Czech": "cs_CZ",
        "Dutch": "nl_XX", "English": "en_XX",
        "Estonian": "et_EE", "Finnish": "fi_FI",
        "French": "fr_XX", "Galician": "gl_ES",
        "Georgian": "ka_GE", "German": "de_DE",
        "Gujarati": "gu_IN", "Hebrew": "he_IL",
        "Hindi": "hi_IN", "Indonesian": "id_ID",
        "Italian": "it_IT", "Japanese": "ja_XX",
        "Kazakh": "kk_KZ", "Khmer": "km_KH",
        "Korean": "ko_KR", "Latvian": "lv_LV",
        "Lithuanian": "lt_LT", "Macedonian": "mk_MK",
        "Malayalam": "ml_IN", "Marathi": "mr_IN",
        "Mongolian": "mn_MN", "Nepali": "ne_NP",
        "Pashto": "ps_AF", "Persian": "fa_IR",
        "Polish": "pl_PL", "Portuguese": "pt_XX",
        "Romanian": "ro_RO", "Russian": "ru_RU",
        "Sinhala": "si_LK", "Slovene": "sl_SI",
        "Spanish": "es_XX", "Swahili": "sw_KE",
        "Swedish": "sv_SE", "Tagalog": "tl_XX",
        "Tamil": "ta_IN", "Telugu": "te_IN",
        "Thai": "th_TH", "Turkish": "tr_TR",
        "Ukrainian": "uk_UA", "Urdu": "ur_PK",
        "Vietnamese": "vi_VN", "Xhosa": "xh_ZA"
        }

        return language_mapping[tgt_lang_str]

    def translate(self, original_name, src_lang, tgt_lang_str) -> str:
        """Перевод названия."""
        tgt_lang = self._unpack_tgt_lang_str(tgt_lang_str)
        try:
            # Выполняем перевод
            # translated = self.used_model.translate(original_name, src_lang, tgt_lang)

            self.used_model.tokenizer.src_lang = src_lang
            encoded_lang = self.used_model.tokenizer(original_name, return_tensors="pt")
            generated_tokens = self.used_model.model.generate(
                **encoded_lang,
                forced_bos_token_id=self.used_model.tokenizer.lang_code_to_id[tgt_lang]
            )

            # Определяем, какой декодер использовать (tokenizer или processor)
            if self.used_model.tokenizer:
                translated = self.used_model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            else:
                translated = self.used_model.processor.batch_decode(generated_tokens, skip_special_tokens=True)

            translated = ''.join(translated)
            logger.info(f"Успешный перевод: '{original_name}' -> '{translated}'")
            return translated
        except Exception as e:
            logger.error(f"Ошибка перевода '{original_name}': {e}")
            return original_name
