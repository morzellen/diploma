# src\main.py
import argparse
import sys
import signal
import logging
from typing import Optional

def main() -> Optional[int]:
    """
    Главная функция запуска приложения с обработкой аргументов и ошибок
    """
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description='Запуск интерфейса обработки изображений')
    parser.add_argument('--port', type=int, default=7860, help='Порт для запуска сервера')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Хост для запуска сервера')
    parser.add_argument('--debug', action='store_true', help='Режим отладки')
    args = parser.parse_args()

    # Инициализация логгера
    from core.utils.get_logger import logger
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    try:
        # Проверка зависимостей
        import gradio as gr
        import torch
        from core.ui.gradio_interface import gradio_interface
        
        logger.info("Запуск приложения")
        logger.debug(f"Версия PyTorch: {torch.__version__}")
        logger.debug(f"Версия Gradio: {gr.__version__}")

        # Создание интерфейса
        app = gradio_interface()
        if app is None:
            raise RuntimeError("Не удалось инициализировать интерфейс")

        # Настройка обработчика сигналов
        def graceful_shutdown(signum, frame):
            logger.info("Получен сигнал завершения, останавливаю сервер...")
            if app.server is not None:
                app.server.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

        # Запуск приложения
        logger.info(f"Сервер запущен на http://{args.host}:{args.port}")
        app.launch(
            server_name=args.host,
            server_port=args.port,
            show_error=True,
            debug=args.debug
        )
        return 0

    except ImportError as e:
        logger.critical(f"Отсутствует обязательная зависимость: {e.name}", exc_info=True)
        return 1
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        return 2

if __name__ == "__main__":
    sys.exit(main())
    