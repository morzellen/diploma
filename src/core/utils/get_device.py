import torch

from core.utils.get_logger import logger

# Определяем устройство для модели
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    if device == 'cpu':
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    logger.info(f"Используемое устройство: {device}")
    return device
