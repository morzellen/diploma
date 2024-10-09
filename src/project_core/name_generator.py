import logging
from src.project_core.model_creator import ModelCreator
from PIL import Image, UnidentifiedImageError

class NameGenerator():
    def __init__(self, model_name):
        self.used_model = ModelCreator(model_name)
        self.device = self.used_model.device

    def generate_name(self, image_path, image_name):
        """Генерация наименования изображения"""
        try:
            # Обрабатываем изображение
            inputs = self.process_image(image_path, image_name)

            # Генерируем наименование
            generated_ids = self.used_model.model.generate(pixel_values=inputs.pixel_values, max_length=50)
            
            # Определяем, какой декодер использовать (tokenizer или processor)
            if self.used_model.tokenizer:
                caption = self.used_model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                caption = self.used_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Убираем нежелательные расширения файлов
            unwanted_words = [".", "gif", "гиф", "png", "пнг", "jpg", "jpeg", "джипег"]
            caption_words = caption.split()  # Разделяем наименование на слова
            filtered_caption = " ".join([word for word in caption_words if word.lower() not in unwanted_words])

            logging.info(f"    Сгенерировано наименование для {image_name}: {filtered_caption}")
            return filtered_caption
        except Exception as e:
            logging.error(f"    Ошибка при генерации наименования для {image_name}: {e}")
            raise
    
    def process_image(self, image_path, image_name):
        """Обработка изображения"""
        try:
            image = Image.open(image_path)

            # Конвертация изображения в RGB формат, если необходимо
            if image.mode != 'RGB':
                logging.info(f"    {image_name} конвертирован в RGB")
                image.convert('RGB')
            
            # Обрабатываем изображение с помощью соответствующего процессора
            inputs = self.used_model.processor(images=image, return_tensors="pt").to(self.device)

            logging.info(f"    {image_name} обрабатывается")
            return inputs
        except UnidentifiedImageError:
            logging.error(f"    Невозможно открыть {image_name}")
            raise
        except Exception as e:
            logging.error(f"    Ошибка при обработке {image_name}: {e}")
            raise