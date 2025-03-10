# src\core\ui\gradio_interface.py
import gradio as gr
from typing import List, Optional
from core.utils.get_logger import logger
from core.ui.tabs.renaming_tab import create_renaming_tab
from core.ui.tabs.classification_tab import create_classification_tab

def gradio_interface() -> Optional[gr.TabbedInterface]:
    """
    Инициализация и конфигурация основного интерфейса Gradio
    с обработкой ошибок и логированием
    """
    logger.info("Инициализация Gradio интерфейса")
    
    try:
        # Создание вкладок с обработкой ошибок
        tabs = []
        themes = ['hmb/amethyst', 'soft']  # Резервные темы
        
        try:
            renaming_tab = create_renaming_tab()
            tabs.append(renaming_tab)
            logger.success("Вкладка переименования успешно создана")
        except Exception as e:
            logger.critical(f"Ошибка создания вкладки переименования: {e}", exc_info=True)
            raise

        try:
            classification_tab = create_classification_tab()
            tabs.append(classification_tab)
            logger.success("Вкладка классификации успешно создана")
        except Exception as e:
            logger.critical(f"Ошибка создания вкладки классификации: {e}", exc_info=True)
            raise

        # Выбор темы с fallback
        selected_theme = None
        for theme in themes:
            try:
                gr.themes.ThemeClass.from_hub(theme)
                selected_theme = theme
                logger.debug(f"Тема '{theme}' доступна, используется")
                break
            except Exception:
                logger.warning(f"Тема '{theme}' недоступна, попытка следующей")

        if not selected_theme:
            selected_theme = gr.themes.Default()
            logger.warning("Использована стандартная тема Gradio")

        interface = gr.TabbedInterface(
            tabs,
            tab_names=['Автоматическое переименование фото', 'Автоматическая классификация фото'],
            theme=selected_theme
        )
        
        logger.info(
            f"Интерфейс успешно инициализирован | "
            f"Количество вкладок: {len(tabs)} | "
            f"Тема: {selected_theme if isinstance(selected_theme, str) else 'default'}"
        )
        return interface

    except Exception as e:
        logger.critical(
            f"Критическая ошибка инициализации интерфейса: {e}",
            exc_info=True
        )
        gr.Warning("Произошла критическая ошибка при запуске интерфейса")
        return None