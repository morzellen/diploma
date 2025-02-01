import torch

from core.utils.get_logger import logger

# Определяем устройство для модели
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Используемое устройство: {device}")
