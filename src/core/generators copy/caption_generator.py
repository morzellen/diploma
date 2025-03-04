# src\core\generators\caption_generator.py
import re
from core.utils.get_logger import logger
from core.creators.captioning_model_creator import CaptioningModelCreator
from PIL import Image, UnidentifiedImageError

class CaptionGenerator():
    def __init__(self, captioning_model_name, device):
        self.used_model = CaptioningModelCreator(captioning_model_name, device)
        self.device = device

    def generate_caption(self, photo_path, photo_name):
        """Генерация наименования изображения"""
        # Обрабатываем изображение
        inputs = self.process_photo(photo_path, photo_name)

        # Генерируем наименование
        generated_ids = self.used_model.model.generate(pixel_values=inputs.pixel_values, max_length=50)
        
        # Определяем, какой декодер использовать (tokenizer или processor)
        if self.used_model.tokenizer:
            caption = self.used_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            caption = self.used_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        unwanted_words = [".", "gif", "png", "jpg", "jpeg", "[ unused0 ]"]
        unwanted_patterns = [re.escape(word) for word in unwanted_words]
        pattern = re.compile(r'\b(?:' + '|'.join(unwanted_patterns) + r')\b', re.IGNORECASE)

        filtered_caption = pattern.sub('', caption)

        logger.info(f"Сгенерировано наименование для {photo_name}: {filtered_caption}")
        return filtered_caption

    def process_photo(self, photo_path, photo_name):
        """Обработка изображения"""
        image = Image.open(photo_path)

        # Конвертация изображения в RGB формат, если необходимо
        if image.mode != 'RGB':
            logger.info(f"{photo_name} конвертирован в RGB")
            image.convert('RGB')
        
        # Обрабатываем изображение с помощью соответствующего процессора
        inputs = self.used_model.processor(images=image, return_tensors="pt").to(self.device)

        logger.info(f"{photo_name} обрабатывается")
        return inputs
    