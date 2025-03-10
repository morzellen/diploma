"""Модуль для управления состоянием обработки."""

import threading
from typing import Optional


class ProcessingState:
    """Класс для thread-safe управления состоянием обработки (Singleton)."""
    
    _instance: Optional['ProcessingState'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> 'ProcessingState':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._is_cancelled = False
                cls._instance._state_lock = threading.RLock()
        return cls._instance
    
    @property
    def is_cancelled(self) -> bool:
        """Проверяет флаг отмены операции."""
        with self._state_lock:
            return self._is_cancelled
    
    def set_cancellation_state(self, value: bool) -> None:
        """Устанавливает флаг отмены операции."""
        with self._state_lock:
            self._is_cancelled = value