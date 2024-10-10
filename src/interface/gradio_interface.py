import os
import gradio as gr
from utils.logger import logger
from project_core.process_handler import ProcessHandler

used_handler = ProcessHandler()

def gradio_interface():
    with gr.Blocks(theme='hmb/amethyst') as demo:
        with gr.Row():  # Загрузка файлов
            files = gr.Files(
                file_count="multiple", 
                file_types=["image"], 
                label="Загрузите изображения"
            )

            model_name = gr.Dropdown(
                choices=["GIT-base", "GIT-large", "BLIP-base", "BLIP-large", "ViT+GPT-2"],
                label="Выберите модель",
                value="BLIP-base"
            )

        process_btn = gr.Button("Обработать", size='sm')

        with gr.Row():
            image_outputs = gr.Gallery(label="Предпросмотр изображений")

            def image_preview(files):
                return [(file.name, os.path.basename(file.name)) for file in files]

            files.change(
                image_preview,
                inputs=files,
                outputs=image_outputs
            )

            edit_list = gr.Textbox(
                label="Редактируйте имена",
                lines=10,
                placeholder="Введите новые названия",
            )

        with gr.Row():
            with gr.Column():
                save_dir = gr.Textbox(
                    label="Укажите директорию для сохранения изображений", 
                    lines=1, 
                    placeholder="Введите путь к директории",
                    # value="E:\Downloads\фото"
                )

                save_btn = gr.Button("Сохранить изображения", size='sm')
            with gr.Column():
                saved_results = gr.Textbox(label="Результаты сохранения", lines=10)

                saved_results_clear_btn = gr.ClearButton(
                    components=[saved_results],
                    value="Очистить результаты сохранения",
                    size='sm'
                )

                saved_results_clear_btn.click()

        # Функция обработки изображений
        def process_fn(files, model_name):
            if not files:
                logger.warning("Не загружены файлы для обработки.")
                return "Пожалуйста, загрузите файлы для обработки."

            try:
                paths, suggestions, translated_suggestions = used_handler.handle_images(files, model_name)
            except Exception as e:
                logger.error(f"Ошибка обработки изображений: {e}")
                return f"Ошибка обработки изображений: {e}"

            paths_str = "\n".join([f"{i}) {value}" for i, value in enumerate(paths)])
            suggestions_str = "\n".join([f"{i}) {value}" for i, value in enumerate(suggestions)])
            translated_suggestions_str = "\n".join([f"{i}) {value}" for i, value in enumerate(translated_suggestions)])
            
            logger.info(f"Пути файлов:\n{paths_str}")
            logger.info(f"Предложенные имена {model_name}:\n{suggestions_str}")
            logger.info(f"Предложенные названия на {used_handler.dest_lang} языке:\n{translated_suggestions_str}")

            # Формируем строки вида: (оригинальное имя) переведённое имя
            edit_list = []
            for file, translated_name in zip(files, translated_suggestions):
                original_name = os.path.basename(file.name)
                edit_list.append(f"({original_name}) {translated_name}")

            return "\n".join(edit_list)

        process_btn.click(
            process_fn, 
            inputs=[files, model_name], 
            outputs=[edit_list]
        )

        # Функция сохранения изображений
        def save_fn(edit_list, files, save_dir):
            if not files:
                logger.warning("Не загружены файлы для сохранения.")
                return "Пожалуйста, загрузите файлы перед сохранением."
            if not save_dir:
                logger.warning("Не указана директория для сохранения.")
                return "Пожалуйста, укажите директорию для сохранения изображений."

            # Получаем список переведённых имён
            edited_names = edit_list.split('\n')
            try:
                translated_names = [edited_name.split(") ")[1] for edited_name in edited_names]  # Извлекаем только переведённые имена
            except IndexError:
                logger.error("Ошибка при парсинге имён для сохранения.")
                return "Ошибка в именах. Пожалуйста, проверьте корректность названий."

            image_paths = [file.name for file in files]

            logger.info(f"Сохраняем файлы с именами: {translated_names}")
            logger.info(f"Сохраняем файлы в директорию: {save_dir}")

            try:
                saved_results = used_handler.save_images(translated_names, image_paths, save_dir)
                return "\n".join(saved_results)
            except Exception as e:
                logger.error(f"Ошибка при сохранении изображений: {e}")
                return f"Ошибка сохранения: {e}"

        save_btn.click(
            save_fn, 
            inputs=[edit_list, files, save_dir], 
            outputs=saved_results
        )

    return demo
