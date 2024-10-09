import os
import shutil
import logging
from src.project_core.name_generator import NameGenerator
from src.utils.name_translator import NameTranslator
from src.utils.temp_cleaner import clear_temp

class ProcessHandler:
    def __init__(self, src_lang='en', dest_lang='ru'):
        self.src_lang = src_lang
        self.dest_lang = dest_lang

    # Обработка изображений
    def handle_images(self, image_path, model_name):
        
        suggestions = []
        translated_suggestions = []
        paths = []
        name_count = {}  # Для отслеживания дублирующихся имён

        for i, file in enumerate(image_path):
            image_name = os.path.basename(file.name)
            logging.info(f"\n{i}) Работаем над файлом: {image_name}")

            try:
                used_generator = NameGenerator(model_name)
                suggested_name = used_generator.generate_name(file.name, image_name)  # Получаем новое имя
                logging.info(f"    Имя файла, предложенное моделью {model_name}: {suggested_name}")
                paths.append(file.name)
                suggestions.append(suggested_name)
            except ValueError as e:
                logging.error(f"    Ошибка обработки файла {image_name}: {e}")
                continue

            # Переводим имя файла
            used_translator = NameTranslator(self.src_lang, self.dest_lang)
            translated_name = used_translator.translate(suggested_name)

            # Проверка на дублирующиеся имена
            if translated_name in name_count:
                name_count[translated_name] += 1
                translated_name = f"{translated_name} {name_count[translated_name]}"
            else:
                name_count[translated_name] = 1

            translated_suggestions.append(translated_name)

        return paths, suggestions, translated_suggestions

    # Сохранение изображений после редактирования
    def save_images(self, translated_names, image_paths, save_dir):
        results = []

        # Создаём папку renamed_images, если обрабатывается несколько файлов
        if len(image_paths) > 1:
            save_dir = os.path.join(save_dir, 'renamed_images')
            os.makedirs(save_dir, exist_ok=True)

        for translated_name, image_path in zip(translated_names, image_paths):
            if not os.path.exists(image_path):  # Проверяем, существует ли исходный файл
                logging.error(f"    Ошибка: файл не найден {image_path}")
                results.append(f"Ошибка: файл не найден {image_path}")
                continue

            image_extension = os.path.splitext(image_path)[1]  # Получаем расширение файла
            save_dir = os.path.join(save_dir, translated_name.strip() + image_extension)  # Новый путь с учётом имени и расширения

            try:
                shutil.move(image_path, save_dir)
                logging.info(f"    Файл успешно сохранён: {save_dir}")
                results.append(f"Файл успешно сохранён: {save_dir}")
            except Exception as e:
                logging.error(f"    Ошибка при перемещении файла {image_path}: {e}")
                results.append(f"Ошибка при перемещении файла {image_path}: {e}")

                # Если перемещение не удалось, копируем файл
                try:
                    shutil.copy(image_path, save_dir)
                    os.remove(image_path)  # Удаляем оригинал
                    logging.info(f"    Файл скопирован и сохранён: {save_dir}")
                    results.append(f"Файл скопирован и сохранён: {save_dir}")
                except Exception as e2:
                    logging.error(f"    Ошибка при копировании файла {image_path}: {e2}")
                    results.append(f"Ошибка при копировании файла {image_path}: {e2}")
        
        clear_temp()  # Очищаем временные файлы
        return results  # Возвращаем результаты для отображения
