# src\core\ui\tabs\renaming_tab.py
import gradio as gr

from core.handlers.renaming_handler import RenamingHandler
from core.constants.web import TRANSLATION_LANGUAGES
from core.constants.models import CAPTIONING_MODEL_NAMES, TRANSLATION_MODEL_NAMES
from core.ui.common_utils import (create_processing_tab, create_save_decorator, 
                                  initialize_photo_gallery, update_button_states,
                                  cancel_operation, select_directory, clear_temporary_data
                                  )

def create_renaming_tab():
    with gr.Blocks() as renaming_tab:
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    captioning_model = gr.Dropdown(
                        label="Выберите модель для переименования",
                        choices=list(CAPTIONING_MODEL_NAMES.keys()),
                        value="blip-image-captioning-base",
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
                    label="Директория для сохранения фото с новыми названиями", 
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

                    translated_names_df = gr.Dataframe(
                        headers=["№", "Новое имя"],
                        datatype=["number", "str"],
                        col_count=(2, "fixed"),
                        row_count=(0, "dynamic"),
                        interactive=[False, True],
                        label="Целевые имена",
                        wrap=True,
                        value=[]
                    )

                    # Инициализация галереи через общую функцию
                    photo_tuple.upload(
                        initialize_photo_gallery,
                        inputs=photo_tuple,
                        outputs=[photo_tuple, translated_names_df],
                        show_progress=False
                    )

                    with gr.Column():
                        process_btn = gr.Button("Переименовать фото", size='sm', variant="primary")
                        cancel_btn = gr.Button("Отменить", size='sm', variant="stop", visible=False)

                        @create_processing_tab(RenamingHandler, "переименования")
                        def generic_process_renaming(photo_tuple, captioning_model, translation_model, tgt_lang_str):
                            pass

                        process_event = process_btn.click(
                            fn=lambda is_processing: update_button_states(is_processing, process_btn, cancel_btn),
                            inputs=[gr.State(True)],
                            outputs=[process_btn, cancel_btn]
                        ).then(
                            fn=generic_process_renaming,
                            inputs=[photo_tuple, captioning_model, translation_model, tgt_lang_str],
                            outputs=[translated_names_df, photo_tuple]
                        ).then(
                            fn=lambda is_processing: update_button_states(is_processing, process_btn, cancel_btn),
                            inputs=[gr.State(False)],
                            outputs=[process_btn, cancel_btn]
                        )

                        cancel_btn.click(
                            fn=cancel_operation,
                            inputs=None,
                            outputs=[],
                            cancels=[process_event]
                        )

                        select_dir_btn = gr.Button("Выбрать директорию", size='sm')
                        select_dir_btn.click(
                            fn=lambda current: select_directory("Выберите папку для сохранения фото", current),
                            inputs=[save_dir],
                            outputs=save_dir,
                        )

                        save_btn = gr.Button("Сохранить фото", size='sm')

                        @create_save_decorator(RenamingHandler, "Новое имя", "")
                        def generic_save_renaming(df_data, photo_tuple, save_dir):
                            pass

                        save_btn.click(
                            fn=generic_save_renaming,
                            inputs=[translated_names_df, photo_tuple, save_dir],
                            outputs=[photo_tuple, translated_names_df],  # Указаны выходные компоненты, но обновления не будет
                            show_progress=True
                        ).then(
                            lambda: None,
                            None,
                            None,
                            js="() => {document.querySelector('.progress-bar').style.width = '0%';}"
                        )

                        clear_temp_btn = gr.Button("Очистить временную директорию Gradio", size='sm')
                        clear_temp_btn.click(
                            fn=clear_temporary_data,
                            inputs=None,
                            outputs=[]
                        )

                        clear_components_btn = gr.Button("Cбросить компоненты", size='sm')
                        clear_components_btn.click(
                            fn=lambda: ([],[]),
                            inputs=None,
                            outputs=[photo_tuple, translated_names_df]
                        )

        return renaming_tab
