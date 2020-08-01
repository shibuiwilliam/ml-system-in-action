import os
import json
import logging

from src.api_composition_proxy import helpers

logger = logging.getLogger(__name__)


class _ServiceConfigurations():
    services = {k: v for k, v in os.environ.items() if k.lower().startswith('service_')}
    urls = {k: helpers.url_builder(v) for k, v in services.items()}
    customized_redirect_map = None
    if 'CUSTOMIZED_REDIRECT_MAP' in os.environ.keys():
        customized_redirect_map = json.loads(os.getenv('CUSTOMIZED_REDIRECT_MAP'))


class _FastAPIConfigurations():
    title = os.getenv('FASTAPI_TITLE', 'ServingProxy')
    description = os.getenv(
        'FASTAPI_DESCRIPTION',
        'machine learning system serving proxy')
    version = os.getenv('FASTAPI_VERSION', '0.1')
    app_name = os.getenv(
        'APP_NAME',
        'src.api_composition_proxy.apps.proxy:app')


ServiceConfigurations = _ServiceConfigurations()
FastAPIConfigurations = _FastAPIConfigurations()

logger.info(f'fastapi configurations: {ServiceConfigurations.__dict__}')
logger.info(f'service configurations: {FastAPIConfigurations.__dict__}')
