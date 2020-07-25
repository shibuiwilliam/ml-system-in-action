import enum


class PLATFORM_ENUM(enum.Enum):
    DOCKER_COMPOSE = 'docker_compose'
    KUBERNETES = 'kubernetes'
    TEST = 'test'
