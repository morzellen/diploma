import logging

FORMAT = '\t%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger('main')
handler = logging.FileHandler("log.log")
handler.setFormatter(logging.Formatter(FORMAT))
logger.addHandler(handler)
logger.setLevel(logger.INFO)
