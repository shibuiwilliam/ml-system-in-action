import logging

logger = logging.getLogger(__name__)


def url_builder(hostname: str, path: str):
    if not (hostname.startswith('http://') or hostname.startswith('https://')):
        hostname = f'http://{hostname}'
    if hostname.endswith('/'):
        hostname = f'{hostname}{path}'
    else:
        hostname = f'{hostname}/{path}'
    return hostname