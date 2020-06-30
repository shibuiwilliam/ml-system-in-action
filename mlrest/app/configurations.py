import os
from app.constants import PHYSICAL_SAVE_DATA


class _Configurations():
    fallback_model_filname = 'iris_svc.pkl'
    model_filename = os.getenv('IRIS_MODEL', fallback_model_filname)
    physical_save_data = os.getenv(
        'PHYSICAL_SAVE_DATA',
        PHYSICAL_SAVE_DATA.SAVE)


configurations = _Configurations()
