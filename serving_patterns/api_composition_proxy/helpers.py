import logging

logger = logging.getLogger(__name__)

def path_builder(url: str, path: str) -> str:
    if path.startswith('/'):
        path = path[1:]
    if url.endswith('/'):
        url = f'{url}{path}'
    else:
        url = f'{url}/{path}'
    return url


def url_builder(hostname: str, https: bool = False) -> str:
    if not (hostname.startswith('http://') or hostname.startswith('https://')):
        hostname = f'https://{hostname}' if https else f'http://{hostname}'
    return hostname


def url_path_builder(hostname: str, path: str) -> str:
    hostname = url_builder(hostname, path)
    url = path_builder(hostname, path)
    return url
