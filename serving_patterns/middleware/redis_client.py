import redis

from configurations.configurations import _RedisCacheConfigurations


redis_client = redis.Redis(
    host=_RedisCacheConfigurations().cache_host,
    port=_RedisCacheConfigurations().cache_port,
    db=_RedisCacheConfigurations().redis_db,
    decode_responses=_RedisCacheConfigurations().redis_decode_responses)
