import os
import gradio as gr
from core.handlers.process_handler import RenamingProcessHandler
from core.utils.get_logger import logger
from core.constants.models import SEGMENTATION_MODEL_NAMES, TRANSLATION_MODEL_NAMES
from core.constants.web import TRANSLATION_LANGUAGES

def create_classification_tab():
    with gr.Blocks() as classification_tab:
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    segmentation_model_name = gr.Dropdown(
                        choices=list(SEGMENTATION_MODEL_NAMES.keys()),
                        label="Выберите модель для классификации",
                        value="blip-image-captioning-base",
                        allow_custom_value=True
                    )

                    translation_model_name = gr.Dropdown(
                        choices=list(TRANSLATION_MODEL_NAMES.keys()),
                        label="Выберите модель для перевода",
                        value="mbart-large-50-many-to-many-mmt",
                        allow_custom_value=True
                    )

                    tgt_lang_str = gr.Dropdown(
                        label='Выберите целевой язык перевода',
                        choices=list(TRANSLATION_LANGUAGES.keys()),
                        value='Russian',
                        allow_custom_value=True
                    )



                with gr.Row():
                    photo_tuple = gr.Gallery(
                        label="Загрузите фото",
                        format="jpeg",
                        #file_types=["image"],
                        height=None,
                        scale=1,
                        interactive=True,
                    )
                    def _give_caption(photo_tuple):
                        if not photo_tuple:
                            return []
                        
                        # Возвращаем список кортежей (путь файла, индексированный basename)
                        return [(photo_tuple[0], f'{i}) {os.path.basename(photo_tuple[0])}') for i, photo_tuple in enumerate(photo_tuple, start=1)]
                    photo_tuple.upload(
                        _give_caption,
                        inputs=photo_tuple,
                        outputs=photo_tuple,
                        show_progress=False
                    )
                    translated_originals_str = gr.Textbox(
                        label="Целевые имена",
                        placeholder="Введите новые названия",
                        max_length=None,
                        scale=2,
                        info='Редактируйте, если необходимо, не трогая "i) "'
                    )

                process_btn = gr.Button("Переименовать фото", size='sm')
                # Функция обработки изображений
                def process_fn(photo_tuple, captioning_model_name, translation_model_name, tgt_lang_str):
                    if not photo_tuple:
                        logger.warning("Не загружены фото для обработки.")
                        return 'Не загружены фото для обработки.'
                    
                    try:
                        originals, translated_originals = RenamingProcessHandler.handle_photo(photo_tuple, captioning_model_name, translation_model_name, "en_XX", tgt_lang_str)
                    except Exception as e:
                        logger.error(f"Ошибка обработки фото: {e}")
                        return f"Ошибка обработки фото: {e}"

                    originals_str = "\n".join([f"{i}) {original}" for i, original in enumerate(originals, start=1)])
                    logger.info(f"Предложенные имена {captioning_model_name}:\n{originals_str}")

                    translated_originals_str = "\n".join([f"{i}) {translated_original}" for i, translated_original in enumerate(translated_originals, start=1)])
                    logger.info(f"Предложенные названия на {tgt_lang_str} языке:\n{translated_originals_str}")

                    return translated_originals_str
                process_btn.click(
                    process_fn, 
                    inputs=[photo_tuple, segmentation_model_name, translation_model_name, tgt_lang_str], 
                    outputs=translated_originals_str
                )

                with gr.Row():
                    with gr.Column():
                        saved_results = gr.Textbox(label="Результаты сохранения", max_length=None)
                    with gr.Column():
                        save_dir = gr.Textbox(
                            label="Укажите директорию для сохранения фото", 
                            placeholder="Введите путь к директории",
                            value="C:\Flash\pics",
                            max_length=None
                        )
                        save_btn = gr.Button("Сохранить фото", size='sm')
                        # Функция сохранения изображений
                        def save_fn(translated_originals_str, photo_tuple, save_dir):
                            if not photo_tuple:
                                logger.warning("Не загружены фото для сохранения.")
                                return 'Не загружены фото для сохранения.'
                            
                            if not save_dir:
                                logger.warning("Не указана директория для сохранения фото.")
                                return "Не указана директория для сохранения фото."
                            
                            try:
                                translated_originals = translated_originals_str.split('\n') # Получаем список переведённых имён
                                tgt_names = [translated_original.split(") ")[1] for translated_original in translated_originals]  # Извлекаем только переведённые имена

                                logger.info(f"Сохраняем фото с именами: {tgt_names}")
                            except IndexError:
                                logger.warning("Ошибка в именах. Пожалуйста, проверьте корректность названий.")
                                return "Ошибка в именах. Пожалуйста, проверьте корректность названий."

                            logger.info(f"Сохраняем фото в директорию: {save_dir}")

                            try:
                                photo_paths = [photo_tuple[0] for photo_tuple in photo_tuple] # Получаем пути фото
                                saved_results = RenamingProcessHandler.save_photo(tgt_names, photo_paths, save_dir) # Сохраняем фото и получаем список путей к ним
                                return "\n".join(saved_results)
                            except Exception as e:
                                logger.error(f"Ошибка сохранения фото: {e}")
                                return f"Ошибка сохранения: {e}"
                        save_btn.click(
                            save_fn, 
                            inputs=[translated_originals_str, photo_tuple, save_dir], 
                            outputs=saved_results
                        )

                        saved_results_clear_btn = gr.ClearButton(
                            components=[saved_results],
                            value="Очистить результаты сохранения",
                            size='sm'
                        )
                        saved_results_clear_btn.click()


#                 with gr.Column(scale=1):
#                     description = gr.Textbox(
#                         label="Описание и Пользовательские инструкции",
#                         value=''' '''
#                     )

        return classification_tab
    