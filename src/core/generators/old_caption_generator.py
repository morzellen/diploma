# src\core\generators\caption_generator.py
# import re
# from core.utils.get_logger import logger
# from core.creators.captioning_model_creator import CaptioningModelCreator
# from PIL import Image, UnidentifiedImageError

# class CaptionGenerator:
#     UNWANTED_WORDS = [".", "gif", "png", "jpg", "jpeg", "[ unused0 ]"]
#     UNWANTED_PATTERN = re.compile(
#         r'\b(?:' + '|'.join(re.escape(word) for word in UNWANTED_WORDS) + r')\b', 
#         flags=re.IGNORECASE
#     )


#     def __init__(self, captioning_model_name, device):
#         self.creator = CaptioningModelCreator(captioning_model_name, device)
#         self.device = device


#     def generate_caption(self, photo_path, photo_name):
#         """Генерация описания с оптимизациями"""
#         try:
#             inputs = self._process_photo(photo_path, photo_name)
#             generated_ids = self.creator.generate(
#                 inputs.pixel_values, 
#                 max_length=50,
#                 num_beams=5  # Оптимизация поиска
#             )
#             caption = self.creator.decode(generated_ids)
#             return self._filter_caption(caption, photo_name)
#         except Exception as e:
#             logger.error(f"Ошибка генерации для {photo_name}: {e}")
#             return ""


#     def _process_photo(self, photo_path, photo_name):
#         """Оптимизированная обработка изображений"""
#         try:
#             with Image.open(photo_path) as img:
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
#                 inputs = self.creator.processor(
#                     images=img, 
#                     return_tensors="pt"
#                 ).to(self.device)
#                 logger.debug(f"Обработано: {photo_name}")
#                 return inputs
#         except Exception as e:
#             logger.error(f"Ошибка обработки {photo_name}: {e}")
#             raise


#     def _filter_caption(self, caption, photo_name):
#         """Фильтрация результата"""
#         filtered = self.UNWANTED_PATTERN.sub('', caption)
#         logger.info(f"Сгенерировано для {photo_name}: {filtered}")
#         return filtered.strip()