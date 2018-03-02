import logging
def get_logger(name, level=logging.DEBUG, file_level=logging.INFO):
  logger = logging.getLogger(name)
  logger.setLevel(level)

  # create a file handler
  handler = logging.FileHandler('../log.log')
  handler.setLevel(file_level)

  # create a logging format
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)

  # add the handlers to the logger
  logger.addHandler(handler)

  return logger



