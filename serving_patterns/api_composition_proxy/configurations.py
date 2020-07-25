import os
import logging

from api_composition_proxy.constants import PLATFORM_ENUM
from api_composition_proxy import helpers

logger = logging.getLogger(__name__)


class _PlatformConfigurations():
    # can be docker_compose or kubernetes
    platform = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER_COMPOSE.value)
    platform = platform if platform in (
        PLATFORM_ENUM.DOCKER_COMPOSE.value,
        PLATFORM_ENUM.KUBERNETES.value) else PLATFORM_ENUM.TEST.value

class _Services():
    services = {k:v for k,v in os.environ.items() if k.lower().startswith('service_')}
    urls = {k:helpers.url_builder(v) for k,v in services.items()}


class _FastAPIConfigurations():
    title = os.getenv('FASTAPI_TITLE', 'ServingProxy')
    description = os.getenv('FASTAPI_DESCRIPTION', 'machine learning system serving proxy')
    version = os.getenv('FASTAPI_VERSION', '0.1')
    app_name = os.getenv('APP_NAME', 'api_composition_proxy.apps.proxy:app')


class _CacheConfigurations():
    cache_host = os.getenv('CACHE_HOST', 'redis')
    cache_port = int(os.getenv('CACHE_PORT', 6379))
    queue_name = os.getenv('QUEUE_NAME', 'queue')


class _RedisCacheConfigurations(_CacheConfigurations):
    redis_db = int(os.getenv('REDIS_DB', 0))
    redis_decode_responses = bool(os.getenv('REDIS_DECODE_RESPONSES', True))


logger.info(f'fastapi configurations: {_FastAPIConfigurations.__dict__}')
logger.info(f'service configurations: {_Services.__dict__}')
logger.info(f'platform configurations: {_PlatformConfigurations.__dict__}')
logger.info(f'redis cache configurations: {_RedisCacheConfigurations.__dict__}')

