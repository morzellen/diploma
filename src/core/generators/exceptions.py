# src/core/exceptions.py

class ImageProcessingError(Exception):
    """Исключение, возникающее при ошибках обработки изображений."""
    pass

class CaptionGenerationError(Exception):
    """Исключение, возникающее при ошибках генерации описания."""
    pass

class SegmenatationGenerationError(Exception):
    """Исключение, возникающее при ошибках генерации описания."""
    pass