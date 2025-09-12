import logging


def setup_logger(name="AI-Powered RAG Guitar Assistant"):
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch= logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] --- [%(message)s]")
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger


logger = setup_logger()

# logger.info("logger process started")
# logger.debug("debugging!")
# logger.error("failed to load")
# logger.critical("critical message")









