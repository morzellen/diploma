import os
import sys

import gradio as gr
from PyQt5.QtWidgets import QApplication, QFileDialog

from core.utils.get_logger import logger
from core.handlers.classification_handler import ClassificationHandler
from core.constants.web import TRANSLATION_LANGUAGES
from core.constants.models import SEGMENTATION_MODEL_NAMES, TRANSLATION_MODEL_NAMES

# Глобальная переменная для отслеживания состояния отмены
processing_cancelled = False

def create_classification_tab():
    with gr.Blocks() as classification_tab:
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    segmentation_model = gr.Dropdown(
                        label="Выберите модель для классификации",
                        choices=list(SEGMENTATION_MODEL_NAMES.keys()),
                        value="Florence-2-base",
                        allow_custom_value=True
                    )
                    translation_model = gr.Dropdown(
                        label="Выберите модель для перевода",
                        choices=list(TRANSLATION_MODEL_NAMES.keys()),
                        value="mbart-large-50-many-to-many-mmt",
                        allow_custom_value=True
                    )
                    tgt_lang_str = gr.Dropdown(
                        label='Выберите целевой язык перевода',
                        choices=list(TRANSLATION_LANGUAGES.keys()),
                        value='Russian',
                        allow_custom_value=True
                    )

                save_dir = gr.Textbox(
                    label="Директория для сохранения классов с фото", 
                    value="../classification_results",
                    max_length=None,
                    interactive=False
                )

                with gr.Row():
                    photo_tuple = gr.Gallery(
                        label="Загрузите фото",
                        format="jpeg",
                        height="auto",
                        scale=1,
                        interactive=True,
                        columns=3,
                        object_fit="cover",
                        show_download_button=False,
                        show_share_button=False,
                        show_fullscreen_button=False,
                    )

                    classes_df = gr.Dataframe(
                        headers=["№", "Класс"],
                        datatype=["number", "str"],
                        col_count=(2, "fixed"),
                        row_count=(0, "dynamic"),
                        interactive=[False, True],
                        label="Классы изображений",
                        wrap=True,
                        value=[]
                    )

                    def _initialize_gallery(photo_tuple):
                        if not photo_tuple:
                            return [], []
                        
                        photo_list = [(photo_tuple[0], f'{i}) {os.path.basename(photo_tuple[0])}') 
                                    for i, photo_tuple in enumerate(photo_tuple, start=1)]
                        
                        df_data = [[i, ""] for i in range(1, len(photo_tuple) + 1)]
                        
                        return photo_list, df_data

                    photo_tuple.upload(
                        _initialize_gallery,
                        inputs=photo_tuple,
                        outputs=[photo_tuple, classes_df],
                        show_progress=False
                    )

                    with gr.Column():
                        classify_btn = gr.Button("Классифицировать", size='sm', variant="primary")
                        cancel_btn = gr.Button("Отменить", size='sm', variant="stop", visible=False)

                        def process_fn(photo_tuple, segmentation_model, translation_model, tgt_lang_str, progress=gr.Progress()):
                            global processing_cancelled
                            processing_cancelled = False
                            
                            if not photo_tuple:
                                logger.warning("Не загружены фото для обработки.")
                                gr.Warning("Не загружены фото для обработки")
                                return [], []
                            

                            progress(0, desc="Начало классификации...")
                            gr.Info("Начало классификации фотографий")
                            
                            originals, translated_classes = [], []
                            current_progress = []
                            
                            for result in ClassificationHandler.handle_photo_generator(
                                photo_tuple, 
                                segmentation_model, 
                                translation_model, 
                                "en_XX", 
                                tgt_lang_str,
                                progress,
                                lambda: processing_cancelled
                            ):
                                if isinstance(result, tuple):
                                    original, translated = result
                                    originals.append(original)
                                    translated_classes.append(translated)
                                    current_progress = [[i+1, name] for i, name in enumerate(translated_classes)]
                                    yield current_progress, photo_tuple
                                else:
                                    if "Ошибка" in result:
                                        gr.Warning(result)
                                    yield current_progress, photo_tuple

                            if processing_cancelled:
                                gr.Warning("Операция была отменена пользователем")
                                yield current_progress, photo_tuple
                                return [], []


                            progress(1.0, desc="Готово!")
                            gr.Info("Классификация завершена")
                            yield current_progress, photo_tuple

                        def toggle_buttons(is_processing):
                            return {
                                classify_btn: gr.update(interactive=not is_processing),
                                cancel_btn: gr.update(visible=is_processing)
                            }

                        process_event = classify_btn.click(
                            fn=toggle_buttons,
                            inputs=[gr.State(True)],
                            outputs=[classify_btn, cancel_btn],
                        ).then(
                            process_fn,
                            inputs=[photo_tuple, segmentation_model, translation_model, tgt_lang_str],
                            outputs=[classes_df, photo_tuple],
                        ).then(
                            toggle_buttons,
                            inputs=[gr.State(False)],
                            outputs=[classify_btn, cancel_btn]
                        )

                        def cancel_fn():
                            global processing_cancelled
                            processing_cancelled = True
                            logger.info("Запрошена отмена операции")
                            gr.Info("Запрошена отмена операции")
                            return None

                        cancel_btn.click(
                            fn=cancel_fn,
                            inputs=None,
                            outputs=None,
                            cancels=[process_event]
                        )

                        select_dir_btn = gr.Button("Выбрать директорию", size='sm')
                
                        def select_folder():
                            app = QApplication(sys.argv)
                            folder_path = QFileDialog.getExistingDirectory(None, "Выберите папку для сохранения классов")
                            app.quit()
                            return folder_path
                        
                        select_dir_btn.click(
                            fn=select_folder,
                            inputs=None,
                            outputs=save_dir,
                        )

                        save_btn = gr.Button("Сохранить классы", size='sm')

                        def save_fn(df_data, photo_tuple, save_dir, progress=gr.Progress()):  # Добавили прогресс-бар
                            if not photo_tuple:
                                logger.warning("Не загружены фото для сохранения.")
                                gr.Warning("Не загружены фото для сохранения")
                                return [], []  # Корректный возврат пустых списков

                            try:
                                progress(0, desc="Начало обработки...")
                                
                                # Проверка DataFrame
                                if df_data.empty:
                                    logger.warning("DataFrame пустой")
                                    gr.Warning("Нет данных для сохранения")
                                    return [], []

                                # Получаем данные из DataFrame по именам столбцов
                                class_names = df_data["Класс"].astype(str).str.strip().tolist()
                                indices = df_data["№"].astype(int).tolist()

                                photo_pairs = []
                                has_empty_classes = False
                                
                                # Обрабатываем каждую запись
                                for idx, class_name in zip(indices, class_names):
                                    try:
                                        photo_idx = idx - 1  # Конвертация в 0-based индекс
                                        
                                        if 0 <= photo_idx < len(photo_tuple):
                                            # Получаем путь к файлу из Gallery
                                            photo_item = photo_tuple[photo_idx]
                                            photo_path = photo_item[0] if isinstance(photo_item, tuple) else photo_item
                                            
                                            # Обработка пустых классов
                                            if not class_name:
                                                class_name = f"Неизвестный_класс_{idx}"
                                                has_empty_classes = True
                                                
                                            photo_pairs.append((photo_path, class_name))
                                            
                                    except (IndexError, ValueError) as e:
                                        logger.warning(f"Ошибка обработки строки {idx}: {e}")
                                        continue

                                if not photo_pairs:
                                    logger.warning("Нет валидных данных для сохранения")
                                    gr.Warning("Нет данных для сохранения")
                                    return [], []

                                if has_empty_classes:
                                    gr.Info("Некоторые классы были пустыми и заменены")

                                # Сохраняем с прогрессом
                                progress(0.3, desc="Сохранение файлов...")
                                paths, classes = zip(*photo_pairs)
                                saved_results = ClassificationHandler.save_photo(classes, paths, save_dir)
                                
                                # Статистика
                                success_count = sum(1 for res in saved_results if "успешно" in res.lower())
                                progress(0.8, desc="Формирование отчёта...")
                                
                                if success_count == len(photo_pairs):
                                    gr.Info(f"Успешно сохранено {success_count} файлов")
                                else:
                                    gr.Warning(f"Проблемы при сохранении: {len(photo_pairs)-success_count} ошибок")

                                progress(1.0, desc="Готово!")
                                return [], []  # Корректный возврат

                            except Exception as e:
                                logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
                                gr.Warning(f"Ошибка сохранения: {str(e)}")
                                progress(1.0, desc="Ошибка!")
                                return [], []

                        save_btn.click(
                            fn=save_fn, 
                            inputs=[classes_df, photo_tuple, save_dir], 
                            outputs=[photo_tuple, classes_df]
                        ).then(
                            lambda: None,  # Пустая функция для сброса progress bar
                            None,
                            None,
                            js="() => {document.querySelector('.progress-bar').style.width = '0%';}"  # Сброс progress bar через JavaScript
                        )

        return classification_tab
    