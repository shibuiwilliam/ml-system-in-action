import os
import logging

logger = logging.getLogger(__name__)


class Services():
    services = {k:v for k,v in os.environ.items() if k.lower().startswith('service_')}


logger.info(f'service configurations: {Services.__dict__}')
