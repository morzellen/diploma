import os
import sys

import gradio as gr
from PyQt5.QtWidgets import QApplication, QFileDialog

from core.utils.get_logger import logger
from core.handlers.renaming_handler import RenamingHandler
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
                    label="Директория для сохранения фото с новыми названиями", 
                    value="../renaming_results",
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
                        row_count=(0, "dynamic"),
                        interactive=[False, True],  # Первый столбец нередактируемый
                        label="Целевые имена",
                        wrap=True,
                        value=[]  # Изначально пустой датафрейм
                    )

                    def _give_caption(photo_tuple):
                        if not photo_tuple:
                            # Очищаем датафрейм при отсутствии фотографий
                            return [], []
                        
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
                            
                            for result in RenamingHandler.handle_photo_generator(
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
                                    yield current_progress, photo_tuple
                                else:  # Получено сообщение о статусе
                                    if "Ошибка" in result:
                                        gr.Warning(result)
                                    yield current_progress, photo_tuple

                            if processing_cancelled:
                                gr.Warning("Операция была отменена пользователем")
                                yield current_progress, photo_tuple
                                return [], None

                            progress(0.8, desc="Форматирование результатов...")
                            
                            originals_str = "\n".join([f"{i}) {original}" for i, original in enumerate(originals, start=1)])
                            logger.info(f"Предложенные имена {captioning_model_name}:\n{originals_str}")

                            translated_originals_str = "\n".join([f"{i}) {translated}" 
                                                               for i, translated in enumerate(translated_originals, start=1)])
                            logger.info(f"Предложенные названия на {tgt_lang_str} языке:\n{translated_originals_str}")

                            progress(1.0, desc="Готово!")
                            gr.Info("Обработка фотографий завершена")
                            yield current_progress, photo_tuple

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
                        def save_fn(df_data, photo_tuple, save_dir, progress=gr.Progress()):
                            if not photo_tuple:
                                logger.warning("Не загружены фото для сохранения.")
                                gr.Warning("Не загружены фото для сохранения")
                                return [], []
                            
                            try:
                                logger.info(f"DataFrame data: {df_data}")
                                logger.info(f"Photo tuple: {photo_tuple}")
                                
                                progress(0, desc="Подготовка к сохранению...")
                                
                                photo_pairs = []
                                has_empty_names = False
                                
                                # Обрабатываем данные из DataFrame
                                for index, row in df_data.iterrows():
                                    if not isinstance(row['№'], (int, float)):
                                        continue
                                        
                                    logger.info(f"Processing row: {row}")
                                    try:
                                        idx = int(row['№']) - 1
                                        new_name = str(row['Новое имя']).strip()
                                        
                                        if 0 <= idx < len(photo_tuple):
                                            photo = photo_tuple[idx]
                                            photo_path = photo[0] if isinstance(photo, tuple) else photo
                                            
                                            # Если имя пустое, используем индекс
                                            if not new_name:
                                                new_name = str(idx + 1)  # +1 для человекочитаемой нумерации
                                                has_empty_names = True
                                                
                                            photo_pairs.append((photo_path, new_name))
                                            logger.info(f"Added pair: {photo_path} -> {new_name}")
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"Ошибка обработки строки {row}: {e}")
                                        continue
                                
                                if not photo_pairs:
                                    logger.warning("Нет данных для сохранения")
                                    gr.Warning("Нет данных для сохранения")
                                    return [], []
                                
                                if has_empty_names:
                                    gr.Info("Некоторые имена были пустыми и заменены на индексы")
                                
                                progress(0.3, desc="Сохранение файлов...")
                                
                                paths, names = zip(*photo_pairs)
                                logger.info(f"Сохраняем фото:\nПути: {paths}\nИмена: {names}")
                                
                                saved_results = RenamingHandler.save_photo(names, paths, save_dir)
                                success_count = sum(1 for result in saved_results if "успешно" in result.lower())
                                
                                progress(0.8, desc="Завершение операции...")
                                
                                if success_count == len(photo_pairs):
                                    gr.Info("Все файлы успешно сохранены")
                                else:
                                    gr.Warning(f"Сохранено {success_count} из {len(photo_pairs)} файлов")
                                
                                progress(1.0, desc="Готово!")
                                return [], []
                                
                            except Exception as e:
                                logger.error(f"Ошибка при сохранении: {str(e)}")
                                gr.Warning(f"Ошибка при сохранении: {str(e)}")
                                progress(1.0, desc="Ошибка!")
                                return [], []

                        save_btn.click(
                            fn=save_fn,
                            inputs=[translated_names_df, photo_tuple, save_dir],
                            outputs=[photo_tuple, translated_names_df],  # Обновляем оба компонента
                            show_progress=True
                        ).then(
                            lambda: None,  # Пустая функция для сброса progress bar
                            None,
                            None,
                            js="() => {document.querySelector('.progress-bar').style.width = '0%';}"  # Сброс progress bar через JavaScript
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
    