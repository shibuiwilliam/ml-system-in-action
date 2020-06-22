from fastapi.logger import logger
import logging

gunicorn_error_logger = logging.getLogger('gunicorn.error')
gunicorn_logger = logging.getLogger('gunicorn')
uvicorn_access_logger = logging.getLogger('uvicorn.access')
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
logger.handlers = uvicorn_access_logger.handlers
logger.setLevel(gunicorn_logger.level)

logger.info("Initialized logger")

