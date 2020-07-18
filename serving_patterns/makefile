absolutePath := $(shell pwd)
dockerfileDir := dockerfiles

imageName := serving_patterns_api
imageVersion := latest
containerName := serving_patterns_api

testContainerFile := TestDockerfile
testImageVersion := test
testContainerName := test

trainIrisContainerFile := Dockerfile_train_iris
trainIrisImageName := serving_patterns_train_iris
trainIrisImageVersion := latest
trainIrisContainerName := serving_patterns_train_iris

extractImagenetResnet50ContainerFile := Dockerfile_extract_imagenet_resnet50
extractImagenetResnet50ImageName := serving_patterns_extract_imagenet_resnet50
extractImagenetResnet50ImageVersion := latest
extractImagenetResnet50ContainerName := serving_patterns_extract_imagenet_resnet50

composeBuild := docker-compose build
composeUp := docker-compose up -d
composeDown := docker-compose down

requirements_api := requirements_api.txt
pipFreeze := pip freeze > requirements/$(requirements_api)
pipInstall := pip install -r requirements/$(requirements_api)

pytest := pytest -v -s ./tests/
pytestD := docker build \
				-t $(imageName):$(testImageVersion) \
				-f $(dockerfileDir)/$(testContainerFile) . && \
 		   docker run --rm \
				--name $(testContainerName) \
				$(imageName):$(testImageVersion) $(pytest)

testRequest := ./scripts/test_request.sh

trainIris := docker build \
				-t $(trainIrisImageName):$(trainIrisImageVersion) \
				-f $(dockerfileDir)/$(trainIrisContainerFile) . && \
 			 docker run --rm \
				--name $(trainIrisContainerName) \
				-v $(absolutePath)/app/ml/models/:/serving_patterns/app/ml/models/ \
				$(trainIrisImageName):$(trainIrisImageVersion)

extractImagenetResnet50 := docker build \
				-t $(extractImagenetResnet50ImageName):$(extractImagenetResnet50ImageVersion) \
				-f $(dockerfileDir)/$(extractImagenetResnet50ContainerFile) . && \
 			 docker run --rm \
				--name $(extractImagenetResnet50ContainerName) \
				-v $(absolutePath)/app/ml/models/:/serving_patterns/app/ml/models/ \
				$(extractImagenetResnet50ImageName):$(extractImagenetResnet50ImageVersion)

.PHONY: clean pip_freeze compose_build compose_up compose_down test test_d train_iris test_request extract_imagenet_resnet50
pip_freeze:
	$(pipFreeze)

clean:
	$(composeDown)

compose_build:
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

extract_imagenet_resnet50:
	$(extractImagenetResnet50)
