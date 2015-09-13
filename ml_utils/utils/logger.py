__author__ = 'euclides'

SENTRY_SET = False
NO_FILE = False

import traceback
import os
import platform
from datetime import datetime

ROOT = os.path.abspath(".") + "/"
LOG_ROOT = ROOT + "log/"

NODE = platform.node()

import logging

handler_error = None
handler_warning = None
handler_debug = None
handler_console = None
file_date_format = '%Y-%m-%d'

formatter = logging.Formatter(
    '%(asctime)s\t%(levelname)-8s\t%(name)s\t%(message)s', datefmt='%y-%m-%d %H:%M:%S'
)


def add_file_handler(level):
    file_prefix = LOG_ROOT + NODE + '_' + format(datetime.now(), file_date_format)
    name = logging.getLevelName(level)
    file_handler = logging.FileHandler(filename=(file_prefix + '_%s.log' % name), encoding="utf-8")
    file_handler.name = name.lower()
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logging.getLogger().addHandler(file_handler)
    if logging.getLogger().level > level:
        logging.getLogger().setLevel(level)
    return file_handler


def setSentry(url):
    global SENTRY_SET
    if SENTRY_SET:
        return
    try:
        import raven.handlers.logging as rhl
    except:
        SENTRY_SET = False
        return

    client = rhl.Client(url)
    sentry_handler = rhl.SentryHandler(client)
    sentry_handler.setFormatter(formatter)
    sentry_handler.setLevel(logging.WARNING)
    sentry_handler.name = "sentry"
    logging.getLogger().addHandler(sentry_handler)
    SENTRY_SET = True


def get_loggers(name):
    logger = logging.getLogger(name)
    logd = logger.debug
    logw = logger.warning
    loge = logger.error
    logi = logger.info
    return logi, logd, logw, loge



def setup(only_console=True, log_level=logging.INFO):
    global handler_error, handler_warning, handler_console, handler_debug, NO_FILE
    if handler_debug is not None:
        logging.getLogger().removeHandler(handler_debug)
    if handler_warning is not None:
        logging.getLogger().removeHandler(handler_warning)
    if handler_error is not None:
        logging.getLogger().removeHandler(handler_error)

    handlers = logging.getLogger().handlers
    if len(handlers):
        for h in handlers:
            if isinstance(h, logging.StreamHandler):
                handler_console = h
                break

    if handler_console is None:
        handler_console = logging.StreamHandler()
    if handler_console is not None:
        logging.getLogger().removeHandler(handler_console)
        handler_console.name = "console"
        handler_console.setFormatter(formatter)
        handler_console.setLevel(log_level)
        logging.getLogger().addHandler(handler_console)
        logging.getLogger().setLevel(log_level)

    if not only_console:
        if not os.path.exists(LOG_ROOT):
            try:
                os.makedirs(LOG_ROOT)
                NO_FILE = False
            except:
                traceback.print_exc()
                NO_FILE = True

        if not NO_FILE:
            handler_debug = add_file_handler(logging.DEBUG) if log_level <= logging.DEBUG else None
            handler_warning = add_file_handler(logging.WARNING) if log_level <= logging.WARNING else None
            handler_error = add_file_handler(logging.ERROR) if log_level <= logging.ERROR else None


setup(only_console=True, log_level=logging.INFO)
