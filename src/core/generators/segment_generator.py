from core.utils.get_logger import logger
from core.creators.segmentation_model_creator import SegmentationModelCreator
from PIL import Image, UnidentifiedImageError

class SegmentGenerator():
    def __init__(self, segmentation_model_name, device):
        self.used_model = SegmentationModelCreator(segmentation_model_name, device)
        self.device = device
        