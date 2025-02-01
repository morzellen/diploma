# Запуск приложения
if __name__ == "__main__":
    from core.utils.get_logger import logger
    from core.interface.gradio_interface import gradio_interface

    try:
        gradio_interface().launch()
    except Exception as e:
        logger.error(f"Произошла ошибка при запуске приложения: {e}")
        raise
