# src\core\ui\logic\processing_state.py
"""Модуль для thread-safe управления состоянием обработки с логированием."""

import threading
from typing import Optional
from core.utils.get_logger import logger


class ProcessingState:
    """Класс для управления состоянием обработки с thread-safe доступом (Singleton).
    
    Логирует ключевые изменения состояния и ошибки доступа.
    """
    
    _instance: Optional['ProcessingState'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> 'ProcessingState':
        with cls._lock:
            if cls._instance is None:
                try:
                    cls._instance = super().__new__(cls)
                    cls._instance._is_cancelled = False
                    cls._instance._state_lock = threading.RLock()
                    logger.success("Инициализирован ProcessingState (Singleton)")
                except Exception as e:
                    logger.critical(f"Ошибка создания Singleton: {e}", exc_info=True)
                    raise
        return cls._instance
    
    @property
    def is_cancelled(self) -> bool:
        """Возвращает текущее состояние флага отмены с логированием."""
        try:
            with self._state_lock:
                logger.debug(f"Запрос состояния отмены: {self._is_cancelled}")
                return self._is_cancelled
        except Exception as e:
            logger.error(f"Ошибка доступа к состоянию: {e}", exc_info=True)
            raise

    def set_cancellation_state(self, value: bool) -> None:
        """Устанавливает флаг отмены с валидацией и логированием."""
        try:
            if not isinstance(value, bool):
                error_msg = f"Некорректный тип значения: {type(value)}. Ожидается bool"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            with self._state_lock:
                prev_state = self._is_cancelled
                self._is_cancelled = value
                logger.info(
                    f"Изменение состояния отмены: {prev_state} -> {value} "
                    f"(Поток: {threading.get_ident()})"
                )
                
        except Exception as e:
            logger.error(f"Ошибка установки состояния: {e}", exc_info=True)
            raise

    def reset(self) -> None:
        """Сбрасывает состояние обработки с логированием."""
        try:
            with self._state_lock:
                if self._is_cancelled:
                    self._is_cancelled = False
                    logger.warning("Сброс состояния обработки в исходное положение")
        except Exception as e:
            logger.error(f"Ошибка сброса состояния: {e}", exc_info=True)
            raise
        