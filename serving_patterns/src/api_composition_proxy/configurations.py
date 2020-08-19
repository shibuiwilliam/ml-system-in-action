import os
import json
import logging

from src.api_composition_proxy import helpers
from src.configurations import RedisCacheConfigurations

logger = logging.getLogger(__name__)


class _ServiceConfigurations():
    services = {k: v for k, v in os.environ.items() if k.lower().startswith('service_')}
    urls = {k: helpers.url_builder(v) for k, v in services.items()}
    customized_redirect_map = None
    if 'CUSTOMIZED_REDIRECT_MAP' in os.environ.keys():
        customized_redirect_map = json.loads(os.getenv('CUSTOMIZED_REDIRECT_MAP'))
    enqueue = int(os.getenv('ENQUEUE', 1))


class _APIConfigurations():
    title = os.getenv('API_TITLE', 'ServingProxy')
    description = os.getenv(
        'API_DESCRIPTION',
        'machine learning system serving proxy')
    version = os.getenv('API_VERSION', '0.1')
    app_name = os.getenv(
        'APP_NAME',
        'src.api_composition_proxy.apps.proxy:app')


ServiceConfigurations = _ServiceConfigurations()
APIConfigurations = _APIConfigurations()

logger.info(f'api configurations: {ServiceConfigurations.__dict__}')
logger.info(f'service configurations: {APIConfigurations.__dict__}')
logger.info(f'redis cache configurations: {RedisCacheConfigurations.__dict__}')
