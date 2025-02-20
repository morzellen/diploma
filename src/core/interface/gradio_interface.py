import gradio as gr

from core.utils.get_logger import logger
from core.interface.tabs.renaming_tab import create_renaming_tab
from core.interface.tabs.classification_tab import create_classification_tab

def gradio_interface():
    logger.info("Запуск Gradio интерфейса")

    # theme='hmb/amethyst'
    return gr.TabbedInterface(
        [create_renaming_tab(), create_classification_tab()], 
        ['Автоматическое переименование фото', 'Автоматическая классификация фото'], 
        # theme='earneleh/paris'
    )

