FROM python:3.7-slim

ENV PROJECT_DIR /fastapi/
WORKDIR $PROJECT_DIR
ADD requirements.txt $PROJECT_DIR
RUN pip install -r requirements.txt

COPY ./app/ $PROJECT_DIR/app/
WORKDIR $PROJECT_DIR/app/
RUN chmod +x run.sh
CMD ["./run.sh"]