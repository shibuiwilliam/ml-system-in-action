absolutePath := $(shell pwd)
dockerfileDir := dockerfiles

imageName := serving_patterns_api
imageVersion := latest
containerName := serving_patterns_api

testContainerFile := TestDockerfile
testImageVersion := test
testContainerName := test

trainContainerFile := Dockerfile_train
trainImageName := serving_patterns_train_iris
trainImageVersion := latest
trainContainerName := serving_patterns_train_iris

composeBuild := docker-compose build
composeUp := docker-compose up -d
composeDown := docker-compose down

requirements := requirements.txt
pipFreeze := pip freeze > $(requirements)
pipInstall := pip install -r $(requirements)

pytest := pytest -v -s ./tests/
pytestD := docker build \
				-t $(imageName):$(testImageVersion) \
				-f $(dockerfileDir)/$(testContainerFile) . && \
 		   docker run --rm \
				--name $(testContainerName) \
				$(imageName):$(testImageVersion) $(pytest)

testRequest := ./scripts/test_request.sh

trainIris := docker build \
				-t $(trainImageName):$(trainImageVersion) \
				-f $(dockerfileDir)/$(trainContainerFile) . && \
 			 docker run --rm \
				--name $(trainContainerName) \
				-v $(absolutePath)/app/ml/models/:/serving_patterns/app/ml/models/ \
				$(trainImageName):$(trainImageVersion)

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
	$(trainIris)
