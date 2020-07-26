import os
import logging

from configurations.constants import PLATFORM_ENUM
from api_composition_proxy import helpers

logger = logging.getLogger(__name__)


class _Services():
    services = {k:v for k,v in os.environ.items() if k.lower().startswith('service_')}
    urls = {k:helpers.url_builder(v) for k,v in services.items()}


class _FastAPIConfigurations():
    title = os.getenv('FASTAPI_TITLE', 'ServingProxy')
    description = os.getenv('FASTAPI_DESCRIPTION', 'machine learning system serving proxy')
    version = os.getenv('FASTAPI_VERSION', '0.1')
    app_name = os.getenv('APP_NAME', 'api_composition_proxy.apps.proxy:app')


logger.info(f'fastapi configurations: {_FastAPIConfigurations.__dict__}')
logger.info(f'service configurations: {_Services.__dict__}')

