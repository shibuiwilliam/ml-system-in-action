from fastapi.logger import logger
import logging

gunicorn_error_logger = logging.getLogger('gunicorn.error')
gunicorn_access_logger = logging.getLogger('gunicorn.access')
gunicorn_logger = logging.getLogger('gunicorn')

# uvicorn_access_logger = logging.getLogger('uvicorn.access')
# uvicorn_error_logger = logging.getLogger('uvicorn.error')
# uvicorn_logger = logging.getLogger('uvicorn')

logger.handlers = gunicorn_logger.handlers
# logger.addHandler(uvicorn_logger.handlers)
# logger.addHandler(gunicorn_logger.handlers)
# logger.addHandler(uvicorn_access_logger.handlers)
# logger.addHandler(uvicorn_error_logger.handlers)
# logger.addHandler(gunicorn_access_logger.handlers)
# logger.addHandler(gunicorn_error_logger.handlers)
logger.setLevel(gunicorn_logger.level)

logger.info("Initialized logger")

