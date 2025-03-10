# src\core\ui\gradio_interface.py
import gradio as gr

from core.utils.get_logger import logger
from core.ui.tabs.renaming_tab import create_renaming_tab
from core.ui.tabs.classification_tab import create_classification_tab

def gradio_interface():
    logger.info("Запуск Gradio интерфейса")

    return gr.TabbedInterface(
        [create_renaming_tab(), create_classification_tab()], 
        ['Автоматическое переименование фото', 'Автоматическая классификация фото'], 
        # theme='earneleh/paris'
        theme='hmb/amethyst'
    )

