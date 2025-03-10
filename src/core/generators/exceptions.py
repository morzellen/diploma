# src/core/exceptions.py

class ImageProcessingError(Exception):
    """Исключение, возникающее при ошибках обработки изображений."""
    pass

class CaptionGenerationError(Exception):
    """Исключение, возникающее при ошибках генерации описания."""
    pass

class SegmentationGenerationError(Exception):
    """Исключение, возникающее при ошибках генерации описания."""
    pass

class TranslationGenerationError(Exception):
    """Исключение, возникающее при ошибках генерации перевода."""
    pass