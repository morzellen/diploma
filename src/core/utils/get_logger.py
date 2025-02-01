import os
import logging
from datetime import datetime

FORMAT = '%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s'

logging.basicConfig(format=FORMAT)

log_filename = f"../logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

os.makedirs(os.path.dirname(log_filename), exist_ok=True)

handler = logging.FileHandler(log_filename, encoding='utf-8')
handler.setFormatter(logging.Formatter(FORMAT))

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
