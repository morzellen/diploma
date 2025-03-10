# src\core\ui\tabs\classification_tab.py
import gradio as gr
from core.utils.get_logger import logger
from core.handlers.classification_handler import ClassificationHandler
from core.constants.web import TRANSLATION_LANGUAGES
from core.constants.models import SEGMENTATION_MODEL_NAMES, TRANSLATION_MODEL_NAMES
from core.ui.logic.decorators import create_processing_tab, create_save_decorator
from core.ui.logic.ui_utils import initialize_photo_gallery, update_button_states, select_directory
from core.ui.logic.cancellation import cancel_operation
from core.ui.logic.data_management import clear_temporary_data

def create_classification_tab():
    with gr.Blocks() as classification_tab:
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    segmentation_model = gr.Dropdown(
                        label="Выберите модель для классификации",
                        choices=list(SEGMENTATION_MODEL_NAMES.keys()),
                        value="Florence-2-base",
                    )
                    translation_model = gr.Dropdown(
                        label="Выберите модель для перевода",
                        choices=list(TRANSLATION_MODEL_NAMES.keys()),
                        value="mbart-large-50-many-to-many-mmt",
                    )
                    tgt_lang_str = gr.Dropdown(
                        label='Выберите целевой язык перевода',
                        choices=list(TRANSLATION_LANGUAGES.keys()),
                        value='Russian',
                    )

                save_dir = gr.Textbox(
                    label="Директория для сохранения классов с фото", 
                    value="../results",
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
                        wrap=False,
                        value=[]
                    )

                    # Инициализация галереи через общую функцию
                    photo_tuple.upload(
                        initialize_photo_gallery,
                        inputs=photo_tuple,
                        outputs=[photo_tuple, classes_df],
                        show_progress=False
                    )

                    with gr.Column():
                        classify_btn = gr.Button("Классифицировать", size='sm', variant="primary")
                        cancel_btn = gr.Button("Отменить", size='sm', variant="stop", visible=False)

                        @create_processing_tab(ClassificationHandler, "классификации")
                        def generic_process_classification(photo_tuple, segmentation_model, translation_model, tgt_lang_str):
                            try:
                                logger.info(f"Запуск классификации с моделями: {segmentation_model}/{translation_model}")
                            except Exception as e:
                                logger.critical(f"Ошибка инициализации обработки: {e}", exc_info=True)
                                raise

                        process_event = classify_btn.click(
                            fn=lambda is_processing: update_button_states(is_processing, classify_btn, cancel_btn),
                            inputs=[gr.State(True)],
                            outputs=[classify_btn, cancel_btn]
                        ).then(
                            generic_process_classification,
                            inputs=[photo_tuple, segmentation_model, translation_model, tgt_lang_str],
                            outputs=[classes_df, photo_tuple]
                        ).then(
                            lambda is_processing: update_button_states(is_processing, classify_btn, cancel_btn),
                            inputs=[gr.State(False)],
                            outputs=[classify_btn, cancel_btn],
                            show_progress=False
                        ).then(
                            lambda: gr.Info("Классификация завершена успешно!"),
                            None,
                            None
                        )

                        cancel_btn.click(
                            fn=cancel_operation,
                            inputs=None,
                            outputs=[],
                            cancels=[process_event]
                        )

                        select_dir_btn = gr.Button("Выбрать директорию", size='sm')
                        select_dir_btn.click(
                            fn=lambda current: select_directory("Выберите папку для сохранения классов", current),
                            inputs=[save_dir],
                            outputs=save_dir,
                        ).then(
                            lambda path: gr.Info(f"Выбрана директория: {path}") if path else None,
                            inputs=[save_dir],
                            outputs=None
                        )

                        save_btn = gr.Button("Сохранить классы", size='sm')

                        @create_save_decorator(ClassificationHandler, "Класс", "Неизвестный_класс")
                        def generic_save_classification(df_data, photo_tuple, save_dir):
                            try:
                                logger.info(f"Сохранение результатов в {save_dir}")
                            except Exception as e:
                                logger.error(f"Ошибка подготовки сохранения: {e}", exc_info=True)
                                raise

                        save_btn.click(
                            fn=generic_save_classification,
                            inputs=[classes_df, photo_tuple, save_dir],
                            outputs=[photo_tuple, classes_df],
                            show_progress=True,
                        ).then(
                            lambda: None,
                            None,
                            None,
                            js="() => {document.querySelector('.progress-bar').style.width = '0%';}"
                        ).then(
                            lambda: gr.Info("Сохранение завершено успешно!"),
                            None,
                            None
                        )

                        clear_temp_btn = gr.Button("Очистить временную директорию Gradio", size='sm')
                        clear_temp_btn.click(
                            fn=clear_temporary_data,
                            inputs=None,
                            outputs=[]
                        ).then(
                            lambda: gr.Info("Временные данные очищены"),
                            None,
                            None
                        )
                        
                        clear_components_btn = gr.Button("Cбросить компоненты", size='sm')
                        clear_components_btn.click(
                            fn=lambda: ([],[]),
                            inputs=None,
                            outputs=[photo_tuple, classes_df]
                        ).then(
                            lambda: gr.Info("Компоненты сброшены"),
                            None,
                            None
                        )

        return classification_tab
    