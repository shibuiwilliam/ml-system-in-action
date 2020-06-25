import os
from typing import Dict, Any
import logging
from pydantic import BaseModel
import json

logger = logging.getLogger(__name__)


class SaveDataJob(BaseModel):
    job_id: str
    directory: str
    data: Any
    is_completed: bool = False

    def __call__(self):
        save_data_jobs[self.job_id] = self
        logger.info(f'registered job: {self.job_id} in {self.__class__.__name__}')
        file_path = os.path.join(self.directory, f'{self.job_id}.json')
        with open(file_path, 'w') as f:
            json.dump(self.data, f)
        self.is_completed = True
        logger.info(f'completed save data: {self.job_id}')


save_data_jobs: Dict[str, SaveDataJob] = {}
