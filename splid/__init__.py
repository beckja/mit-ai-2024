import logging


def configure_console_logger():
    logger = logging.getLogger()
    h = logging.StreamHandler()
    f = logging.Formatter('%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] %(message)s')
    h.setFormatter(f)
    logger.addHandler(h)
