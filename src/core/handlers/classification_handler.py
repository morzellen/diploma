import os
import shutil

from core.generators.segment_generator import SegmentGenerator
from core.generators.translation_generator import TranslationGenerator
from core.utils.get_logger import logger
from core.utils.utils import clear_temp
from core.utils.get_device import device

class ClassificationHandler:
    """
    Класс для обработки изображений и их классификации.
    """
    
    @classmethod
    def handle_photo_generator(cls, photo_tuple, segmentation_model_name, translation_model_name, src_lang, tgt_lang_str, 
                             progress=None, check_cancelled=None):
        """Генератор для пошаговой обработки фото с классификацией"""

        used_generator = SegmentGenerator(segmentation_model_name, device)
        used_translator = TranslationGenerator(translation_model_name, device)

        total_photos = len(photo_tuple)
        
        for i, photo_tuple in enumerate(photo_tuple):
            if check_cancelled and check_cancelled():
                logger.info("Операция была отменена пользователем")
                return

            if progress is not None:
                progress(0.2 + (0.6 * i / total_photos), desc=f"Обработка фото {i+1} из {total_photos}...")
            
            photo_path = photo_tuple[0]
            photo_name = os.path.basename(photo_path)
            
            yield f"Обработка файла: {photo_name}"
            
            try:
                # Получаем сегменты изображения
                detections = used_generator.generate_segments(photo_path, photo_name)
                
                # Находим главный объект
                main_object = used_generator.get_main_object(detections)
                logger.info(f"Основной объект на фото {photo_name}: {main_object}")
                
                yield f"Определен основной объект для {photo_name}"
                
                # Переводим название класса
                translated_class = used_translator.translate(main_object, src_lang, tgt_lang_str)

                yield (main_object, translated_class)
                
            except Exception as e:
                logger.error(f"Ошибка обработки файла {photo_name}: {e}")
                yield f"Ошибка обработки файла {photo_name}: {e}"
                continue

    @staticmethod
    def save_photo(class_names, photo_paths, save_dir):
        """Сохранение классифицированных изображений по папкам"""
        saved_results = []

        # Создаём корневую папку для классифицированных фото
        save_dir = os.path.join(save_dir, 'classified_photos')
        os.makedirs(save_dir, exist_ok=True)

        for class_name, photo_path in zip(class_names, photo_paths):
            if not os.path.exists(photo_path):
                logger.warning(f"Ошибка: файл не найден {photo_path}")
                saved_results.append(f"Ошибка: файл не найден {photo_path}")
                continue

            # Создаём папку для класса
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Копируем файл в соответствующую папку
            photo_name = os.path.basename(photo_path)
            save_path = os.path.join(class_dir, photo_name)

            try:
                shutil.move(photo_path, save_path)
                logger.info(f"Файл успешно сохранён: {save_path}")
                saved_results.append(f"Файл успешно сохранён: {save_path}")
            except Exception as e:
                logger.error(f"Ошибка при перемещении файла {photo_path}: {e}")
                saved_results.append(f"Ошибка при перемещении файла {photo_path}: {e}")
        
        clear_temp()  # Очищаем временные файлы
        return saved_results
    