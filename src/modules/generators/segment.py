from modules.utils import logger
from modules.model_creators.segmentation import SegmentationModelCreator
from PIL import Image, UnidentifiedImageError

class SegmentGenerator():
    def __init__(self, captioning_model_name, device):
        self.used_model = SegmentationModelCreator(captioning_model_name, device)
        self.device = device