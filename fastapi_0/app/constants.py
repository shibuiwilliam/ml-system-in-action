def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()
    return property(fget, fset)


class _Constants(object):
    @constant
    def MODEL_DIRECTORY():
        return './ml/models/'

    @constant
    def DATA_DIRECTORY():
        return '/fastapi/app/data/'

    @constant
    def IRIS_DATA_DIRECTORY():
        return '/fastapi/app/data/iris/'

    @constant
    def MODEL_EXTENTIONS():
        return ['pkl', 'h5', 'hdf5']

    @constant
    def IRIS_MODEL():
        return 'iris_svc.pkl'


CONSTANTS = _Constants()
