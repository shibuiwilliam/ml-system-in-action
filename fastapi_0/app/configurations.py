import os


class _Configurations():
    fallback_model_filname = 'iris_svc.pkl'
    model_filename = os.getenv('IRIS_MODEL', fallback_model_filname)


Configurations = _Configurations()
