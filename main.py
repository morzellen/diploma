import logging
from src.interface.gradio_interface import gradio_interface

# Запуск приложения
if __name__ == "__main__":
    try:
        logging.info("Запуск Gradio приложения")  # Логируем запуск приложения
        gradio_interface().launch()  # Запуск Gradio приложения
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")  # Логируем ошибки
        