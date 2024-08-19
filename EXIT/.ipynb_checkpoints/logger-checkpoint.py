import logging


def set_handler_of_logger(__logger, print_log=False, write_log=False, filename='log.txt'):
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    __logger.handlers = []

    if write_log:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        __logger.addHandler(file_handler)

    if print_log:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        __logger.addHandler(stream_handler)

    return __logger

logger = logging.getLogger()