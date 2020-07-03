FROM python:3.7-slim

ENV PROJECT_DIR /fastapi/
WORKDIR ${PROJECT_DIR}
ADD requirements.txt ${PROJECT_DIR}
RUN apt-get -y update && \
    apt-get -y install gcc && \
    pip install -r requirements.txt

COPY ./app/ ${PROJECT_DIR}app/
COPY ./run.sh ${PROJECT_DIR}run.sh
RUN chmod +x run.sh
CMD ['./run.sh']