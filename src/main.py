from modules.utils import logger, get_device
from interface.gradio_interface import gradio_interface

# Запуск приложения
if __name__ == "__main__":
    try:
        device = get_device()
        gradio_interface(device).launch()
    except Exception as e:
        logger.error(f"Произошла ошибка при запуске приложения: {e}")
        raise
