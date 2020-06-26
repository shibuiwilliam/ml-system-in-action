def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()
    return property(fget, fset)


class _Constants(object):
    @constant
    def PLATFORM_DOCKER():
        return 'docker'

    @constant
    def PLATFORM_DOCKER_COMPOSE():
        return 'docker_compose'

    @constant
    def PLATFORM_KUBERNETES():
        return 'kubernetes'

    @constant
    def MODEL_DIRECTORY():
        return './ml/models/'

    @constant
    def DATA_DIRECTORY():
        return '/fastapi/app/data/'

    @constant
    def DATA_FILE_DIRECTORY():
        return '/fastapi/app/data/file/'

    @constant
    def MODEL_EXTENTIONS():
        return ['pkl', 'h5', 'hdf5']

    @constant
    def REDIS_INCREMENTS():
        return 'increments'

    @constant
    def IRIS_MODEL():
        return 'iris_svc.pkl'

    @constant
    def PREDICTION_DEFAULT():
        return -1


CONSTANTS = _Constants()
