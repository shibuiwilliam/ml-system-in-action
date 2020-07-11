absolutePath := $(shell pwd)
dockerfileDir := dockerfiles

imageName := mlrest_api
imageVersion := latest
containerName := mlrest_api

testContainerFile := TestDockerfile
testImageVersion := test
testContainerName := test

composeBuild := docker-compose build
composeUp := docker-compose up -d
composeDown := docker-compose down

requirements := requirements.txt
pipFreeze := pip freeze > $(requirements)
pipInstall := pip install -r $(requirements)

pytest := pytest -v -s ./tests/
pytestD := docker build -t $(imageName):$(testImageVersion) -f $(dockerfileDir)/$(testContainerFile) . && \
 		   docker run --rm --name $(testContainerName) $(imageName):$(testImageVersion) $(pytest)

testRequest := ./tests/test_request.sh

trainIris := PYTHONPATH=$(absolutePath) python app/ml/iris/iris_trainer.py 

.PHONY: clean pip_freeze compose_build compose_up compose_down test test_d train_iris test_request
pip_freeze:
	$(pipFreeze)

clean:
	$(composeDown)

compose_build: pip_freeze
	$(composeBuild)

compose_up:
	$(composeUp)

compose_down:
	$(composeDown)

test:
	$(pytest)

test_d: pip_freeze
	$(pytestD)

test_request:
	$(testRequest)

train_iris:
	$(pipInstall)
	$(trainIris)
