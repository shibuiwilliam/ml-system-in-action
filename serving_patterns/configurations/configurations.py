import os
import logging

from configurations.constants import PLATFORM_ENUM


logger = logging.getLogger(__name__)


class _PlatformConfigurations():
    # can be docker_compose or kubernetes
    platform = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER_COMPOSE.value)
    platform = platform if platform in (
        PLATFORM_ENUM.DOCKER_COMPOSE.value,
        PLATFORM_ENUM.KUBERNETES.value) else PLATFORM_ENUM.TEST.value


class _CacheConfigurations():
    cache_host = os.getenv('CACHE_HOST', 'redis')
    cache_port = int(os.getenv('CACHE_PORT', 6379))
    queue_name = os.getenv('QUEUE_NAME', 'queue')


class _RedisCacheConfigurations(_CacheConfigurations):
    redis_db = int(os.getenv('REDIS_DB', 0))
    redis_decode_responses = bool(os.getenv('REDIS_DECODE_RESPONSES', True))


class _FileConfigurations():
    shared_volume = ''
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        shared_volume = '/shared_volume'
    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        shared_volume = '/mnt'
    else:
        shared_volume = '/tmp'


logger.info(f'platform configurations: {_PlatformConfigurations.__dict__}')
logger.info(f'redis cache configurations: {_RedisCacheConfigurations.__dict__}')
logger.info(f'file configurations: {_FileConfigurations.__dict__}')
