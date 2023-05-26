import logging
import os
from app.config import LOG_DIR


class AppLogger:

    def __init__(self, log_level=logging.INFO, log_dir=LOG_DIR):
        self.log_level = log_level
        self.log_dir = log_dir

    def _configure_logger(self, logger_file, logger):
        logger.setLevel(self.log_level)
        handler = logging.FileHandler(logger_file,mode='w')
        handler.setLevel(self.log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        log_file = os.path.join(self.log_dir, logger_name + '.log')
        self._configure_logger(log_file, logger)
        return logger


APP_LOGGER = AppLogger().get_logger('app')
