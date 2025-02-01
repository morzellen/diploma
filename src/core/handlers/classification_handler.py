# class RenamingProcessHandler:
#     """
#     Класс для обработки изображений и сохранения их с новым именем.

#     """
#     # Обработка изображений
#     def handle_photo(device, photo_tuple, captioning_model_name, translation_model_name, src_lang, tgt_lang_str):
#         originals = []
#         translated_originals = []
#         name_count = {}  # Для отслеживания дублирующихся имён

#         used_generator = CaptionGenerator(captioning_model_name, device)
#         used_translator = TranslationGenerator(translation_model_name, device)

#         for i, photo_tuple in enumerate(photo_tuple):
#             photo_path = photo_tuple[0]  # Извлекаем путь к файлу из кортежа
#             photo_name = os.path.basename(photo_path)
#             logger.info(f"{i}) Работаем над файлом: {photo_name}")

#             try:
#                 original_name = used_generator.generate_caption(photo_path, photo_name)  # Получаем новое имя
#                 logger.info(f"Имя файла, предложенное моделью {captioning_model_name}: {original_name}")
#                 originals.append(original_name)
#             except ValueError as e:
#                 logger.error(f"Ошибка обработки файла {photo_name}: {e}")
#                 continue

#             # Переводим имя файла
#             translated_name = used_translator.translate(original_name, src_lang, tgt_lang_str)

#             # Проверка на дублирующиеся имена
#             if translated_name in name_count:
#                 name_count[translated_name] += 1
#                 translated_name = f"{translated_name} {name_count[translated_name]}"
#             else:
#                 name_count[translated_name] = 1

#             translated_originals.append(translated_name)

#         return originals, translated_originals
        
#     # Сохранение изображений после редактирования
#     def save_photo(tgt_names, photo_paths, save_dir):
#         saved_results = []

#         # Создаём папку renamed_photos, если обрабатывается несколько файлов
#         if len(photo_paths) > 1:
#             save_dir = os.path.join(save_dir, 'renamed_photos')
#             os.makedirs(save_dir, exist_ok=True)

#         for tgt_name, photo_path in zip(tgt_names, photo_paths):
#             if not os.path.exists(photo_path):  # Проверяем, существует ли исходный файл
#                 logger.warning(f"Ошибка: файл не найден {photo_path}")
#                 saved_results.append(f"Ошибка: файл не найден {photo_path}")
#                 continue

#             photo_extension = os.path.splitext(photo_path)[1]  # Получаем расширение файла
#             save_dir_path = os.path.join(save_dir, tgt_name.strip() + photo_extension)  # Новый путь с учётом имени и расширения

#             try:
#                 shutil.move(photo_path, save_dir_path)
#                 logger.info(f"Файл успешно сохранён: {save_dir_path}")
#                 saved_results.append(f"Файл успешно сохранён: {save_dir_path}")
#                 clear_temp()  # Очищаем временные файлы
#             except Exception as e:
#                 logger.error(f"Ошибка при перемещении файла {photo_path}: {e}")
#                 saved_results.append(f"Ошибка при перемещении файла {photo_path}: {e}")
        
#         return saved_results  # Возвращаем результаты для отображения