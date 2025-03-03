import torch

from core.utils.get_logger import logger

# Определяем устройство для модели
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logger.info(f"Используемое устройство: {device}")
    return device
