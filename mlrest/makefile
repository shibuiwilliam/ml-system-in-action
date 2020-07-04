absolutePath := $(shell pwd)
dockerfileDir := dockerfiles

imageName := mlrest_app
imageVersion := latest
containerName := mlrest_app

testContainerFile := TestDockerfile
testImageVersion := test
testContainerName := test

composeBuild := docker-compose build
composeUp := docker-compose up -d
composeDown := docker-compose down

requirements := requirements.txt
pipFreeze := pip freeze > $(requirements)

pytest := pytest -v -s ./tests/
pytestD := docker build -t $(imageName):$(testImageVersion) -f $(dockerfileDir)/$(testContainerFile) . && \
 		   docker run --rm --name $(testContainerName) $(imageName):$(testImageVersion) $(pytest)

trainIris := python app/ml/iris/iris_trainer.py 

.PHONY: clean pip_freeze dirs compose_build compose_up compose_down test test_d train_iris
dirs:
	mkdir -p container_files/logs
	mkdir -p container_files/data
	touch container_files/logs/gunicorn_error.log
	touch container_files/logs/gunicorn_access.log
	touch container_files/logs/uvicorn_error.log
	touch container_files/logs/uvicorn_access.log

pip_freeze:
	$(pipFreeze)

clean:
	$(composeDown)

compose_build: pip_freeze dirs
	$(composeBuild)

compose_up:
	$(composeUp)

compose_down:
	$(composeDown)

test:
	$(pytest)

test_d: pip_freeze dirs
	$(pytestD)

train_iris:
	$(trainIris)
