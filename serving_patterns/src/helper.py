import uuid
from src.middleware.redis_client import redis_client
from src.constants import CONSTANTS, JOB_ID_ENUM
from src.configurations import JobIdConfigurations


def get_uuid():
    return str(uuid.uuid4())


def get_incremental_id():
    redis_client.incr(CONSTANTS.REDIS_INCREMENTS)
    incr = redis_client.get(CONSTANTS.REDIS_INCREMENTS)
    return incr


def get_uuid_incremental_id():
    _uuid = get_uuid()
    _incremental_id = get_incremental_id()
    return f'{_uuid}_{_incremental_id}'


def get_job_id():
    if JobIdConfigurations.job_id_type == JOB_ID_ENUM.UUID.value:
        return get_uuid()
    elif JobIdConfigurations.job_id_type == JOB_ID_ENUM.INCREMENTAL.value:
        return get_incremental_id()
    elif JobIdConfigurations.job_id_type == JOB_ID_ENUM.UUID_INCREMENTAL.value:
        return get_uuid_incremental_id()
    else:
        return get_uuid()
