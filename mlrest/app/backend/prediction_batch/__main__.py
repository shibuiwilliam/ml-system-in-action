import logging

from app.backend.prediction_batch import prediction_batch


logger = logging.getLogger('prediction_batch')
logger.addHandler(logging.StreamHandler())


def main():
    prediction_batch.prediction_loop()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    logger.info('start backend')
    main()
