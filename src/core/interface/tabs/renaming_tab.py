import os
import sys

import gradio as gr
from PyQt5.QtWidgets import QApplication, QFileDialog

from core.utils.get_logger import logger
from core.handlers.process_handler import RenamingProcessHandler
from core.constants.web import TRANSLATION_LANGUAGES
from core.constants.models import CAPTIONING_MODEL_NAMES, TRANSLATION_MODEL_NAMES

# Глобальная переменная для отслеживания состояния отмены
processing_cancelled = False

def create_renaming_tab():
    with gr.Blocks() as renaming_tab:
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    captioning_model_name = gr.Dropdown(
                        label="Выберите модель для переименования",
                        choices=list(CAPTIONING_MODEL_NAMES.keys()),
                        value="blip-image-captioning-base",
                        allow_custom_value=True
                    )
                    translation_model_name = gr.Dropdown(
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
                    label="Директория для сохранения фото", 
                    value="../results",
                    max_length=None,
                    interactive=False
                )

                with gr.Row():
                    photo_tuple = gr.Gallery(
                        label="Загрузите фото",
                        format="jpeg",
                        # file_types=["image"],
                        height="auto",
                        scale=1,
                        interactive=True,
                        columns=3,
                        object_fit="cover",
                        show_download_button=False,
                        show_share_button=False,
                        show_fullscreen_button=False,
                    )

                    translated_names_df = gr.Dataframe(
                        headers=["№", "Новое имя"],
                        datatype=["number", "str"],
                        col_count=(2, "fixed"),
                        row_count=(0, "fixed"),  # Запрещаем добавление строк
                        interactive=[False, True],  # Первый столбец нередактируемый
                        label="Целевые имена",
                        wrap=True,
                        value=None  # Изначально пустой датафрейм
                    )

                    def _give_caption(photo_tuple):
                        if not photo_tuple:
                            # Очищаем датафрейм при отсутствии фотографий
                            return [], None
                        
                        # Возвращаем список кортежей и обновленный датафрейм
                        photo_list = [(photo_tuple[0], f'{i}) {os.path.basename(photo_tuple[0])}') 
                                    for i, photo_tuple in enumerate(photo_tuple, start=1)]
                        
                        # Создаем датафрейм с номерами и пустыми именами
                        df_data = [[i, ""] for i in range(1, len(photo_tuple) + 1)]
                        
                        return photo_list, df_data

                    photo_tuple.upload(
                        _give_caption,
                        inputs=photo_tuple,
                        outputs=[photo_tuple, translated_names_df],
                        show_progress=False
                    )

                    with gr.Column():
                        process_btn = gr.Button("Переименовать фото", size='sm', variant="primary")
                        cancel_btn = gr.Button("Отменить", size='sm', variant="stop", visible=False)

                        def process_fn(photo_tuple, captioning_model_name, translation_model_name, tgt_lang_str, progress=gr.Progress()):
                            global processing_cancelled
                            processing_cancelled = False
                            
                            if not photo_tuple:
                                logger.warning("Не загружены фото для обработки.")
                                gr.Warning("Не загружены фото для обработки")
                                return [], None
                            
                            progress(0, desc="Начало обработки...")
                            gr.Info("Начало обработки фотографий")
                            
                            originals, translated_originals = [], []
                            current_progress = []
                            
                            for result in RenamingProcessHandler.handle_photo_generator(
                                photo_tuple, 
                                captioning_model_name, 
                                translation_model_name, 
                                "en_XX", 
                                tgt_lang_str,
                                progress,
                                lambda: processing_cancelled
                            ):
                                if isinstance(result, tuple):  # Получен результат обработки фото
                                    original, translated = result
                                    originals.append(original)
                                    translated_originals.append(translated)
                                    # Обновляем DataFrame
                                    current_progress = [[i+1, name] for i, name in enumerate(translated_originals)]
                                    yield current_progress
                                else:  # Получено сообщение о статусе
                                    if "Ошибка" in result:
                                        gr.Warning(result)
                                    yield current_progress

                            if processing_cancelled:
                                gr.Warning("Операция была отменена пользователем")
                                yield current_progress
                                return [], None

                            progress(0.8, desc="Форматирование результатов...")
                            
                            originals_str = "\n".join([f"{i}) {original}" for i, original in enumerate(originals, start=1)])
                            logger.info(f"Предложенные имена {captioning_model_name}:\n{originals_str}")

                            translated_originals_str = "\n".join([f"{i}) {translated}" 
                                                               for i, translated in enumerate(translated_originals, start=1)])
                            logger.info(f"Предложенные названия на {tgt_lang_str} языке:\n{translated_originals_str}")

                            progress(1.0, desc="Готово!")
                            gr.Info("Обработка фотографий завершена")
                            yield current_progress, current_progress

                        def toggle_buttons(is_processing):
                            return {
                                process_btn: gr.update(interactive=not is_processing),
                                cancel_btn: gr.update(visible=is_processing)
                            }

                        process_event = process_btn.click(
                            fn=toggle_buttons,
                            inputs=[gr.State(True)],
                            outputs=[process_btn, cancel_btn],
                        ).then(
                            process_fn,
                            inputs=[photo_tuple, captioning_model_name, translation_model_name, tgt_lang_str],
                            outputs=[translated_names_df, photo_tuple],
                        ).then(
                            toggle_buttons,
                            inputs=[gr.State(False)],
                            outputs=[process_btn, cancel_btn]
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
                            folder_path = QFileDialog.getExistingDirectory(None, "Выберите папку для сохранения фото")
                            app.quit()
                            return folder_path
                        
                        select_dir_btn.click(
                            fn=select_folder,
                            inputs=None,
                            outputs=save_dir,
                        )

                        save_btn = gr.Button("Сохранить фото", size='sm')
                        # Функция сохранения изображений
                        def save_fn(df_data, photo_tuple, save_dir):
                            if not photo_tuple:
                                logger.warning("Не загружены фото для сохранения.")
                                gr.Warning("Не загружены фото для сохранения")
                                return None, None
                            
                            try:
                                # Проверяем DataFrame на пустоту и валидность данных
                                if df_data is None or len(df_data) == 0:
                                    logger.warning("DataFrame пустой")
                                    gr.Warning("Нет данных для сохранения")
                                    return None, None
                                
                                # Проверяем, что все имена заполнены
                                tgt_names = [row[1] for row in df_data]
                                if any(not name.strip() for name in tgt_names):
                                    logger.warning("Обнаружены пустые имена файлов")
                                    gr.Warning("Пожалуйста, заполните все имена файлов")
                                    return None, None
                                
                                logger.info(f"Сохраняем фото с именами: {tgt_names}")
                                
                                photo_paths = [photo_tuple[0] for photo_tuple in photo_tuple]
                                saved_results = RenamingProcessHandler.save_photo(tgt_names, photo_paths, save_dir)
                                
                                # Подсчитываем успешные операции и ошибки
                                success_count = 0
                                for result in saved_results:
                                    if "успешно" in result.lower():
                                        success_count += 1
                                    elif "файл не найден" in result.lower():
                                        gr.Warning(result)
                                
                                if success_count == len(photo_paths):
                                    gr.Info("Все файлы успешно сохранены")
                                else:
                                    gr.Warning(f"Сохранено {success_count} из {len(photo_paths)} файлов")

                                return None, None
                                
                            except (IndexError, TypeError) as e:
                                logger.warning(f"Ошибка в именах: {e}")
                                gr.Warning("Ошибка в именах. Пожалуйста, проверьте корректность названий.")
                                return None, None

                        save_btn.click(
                            save_fn, 
                            inputs=[translated_names_df, photo_tuple, save_dir], 
                            outputs=[translated_names_df, photo_tuple]
                        )

        return renaming_tab
                # with gr.Column(scale=1):
                #     description = gr.Textbox(
                #         label="Описание и Пользовательские инструкции",
                #         value='''
                #         - Выберите модель для переименования и перевода.
                #         - Загрузите фотографии.
                #         - Нажмите "Переименовать фото".
                #         - Отредактируйте названия, если необходимо, не трогая "i) ".
                #         - Выберите директорию для сохранения фото.
                #         - Нажмите "Сохранить фото".
                #         '''
                #     )
    