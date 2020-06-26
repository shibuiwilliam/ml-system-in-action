import redis

redis_connector = redis.Redis(host='redis', port=6379, db=0)
