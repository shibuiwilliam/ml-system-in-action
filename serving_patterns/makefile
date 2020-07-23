_absolute_path := $(shell pwd)
_dockerfile_dir := dockerfiles

_image_name := serving_patterns_api
_image_version := latest
_container_name := serving_patterns_api

_test_container_file := Dockerfile_test
_test_image_version := test
_test_container_name := test

_train_iris_container_file := Dockerfile_train_iris
_train_iris_image_name := serving_patterns_train_iris
_train_iris_image_version := latest
_train_iris_container_name := serving_patterns_train_iris

_extract_resnet50_container_file := Dockerfile_extract_resnet50
_extract_resnet50_image_name := serving_patterns_extract_resnet50
_extract_resnet50_image_version := latest
_extract_resnet50_container_name := serving_patterns_extract_resnet50

_extract_inceptionv3_container_file := Dockerfile_extract_inceptionv3
_extract_inceptionv3_image_name := serving_patterns_extract_inceptionv3
_extract_inceptionv3_image_version := latest
_extract_inceptionv3_container_name := serving_patterns_extract_inceptionv3

_inceptionv3_tfs_container_file := Dockerfile_tfserving_inceptionv3
_inceptionv3_tfs_image_name := serving_patterns_inceptionv3_tfs
_inceptionv3_tfs_image_version := latest
_inceptionv3_tfs_container_name := serving_patterns_inceptionv3_tfs

_inceptionv3_api_container_file := Dockerfile_api_inceptionv3
_inceptionv3_api_image_name := serving_patterns_inceptionv3_api
_inceptionv3_api_image_version := latest
_inceptionv3_api_container_name := serving_patterns_inceptionv3_api

_compose_api := docker-compose_api.yml
_compose_build_api := docker-compose -f $(_compose_api) build
_compose_up_api := docker-compose -f $(_compose_api) up -d
_compose_down_api := docker-compose -f $(_compose_api) down

_compose_resnet50 := docker-compose_resnet50.yml
_compose_build_resnet50 := docker-compose -f $(_compose_resnet50) build
_compose_up_resnet50 := docker-compose -f $(_compose_resnet50) up -d
_compose_down_resnet50 := docker-compose -f $(_compose_resnet50) down

_compose_inceptionv3 := docker-compose_inceptionv3.yml
_compose_build_inceptionv3 := docker-compose -f $(_compose_inceptionv3) build
_compose_up_inceptionv3 := docker-compose -f $(_compose_inceptionv3) up -d
_compose_down_inceptionv3 := docker-compose -f $(_compose_inceptionv3) down

_requirements_api_test := requirements_api_test.txt
_pip_freeze_test := pip freeze > requirements/$(_requirements_api_test)
_pip_install_test := pip install -r requirements/$(_requirements_api_test)

_pytest := pytest -v -s ./tests/
_pytest_d := docker build \
				-t $(_image_name):$(_test_image_version) \
				-f $(_dockerfile_dir)/$(_test_container_file) . && \
			docker run --rm \
				--name $(_test_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				-e PROFILE=0 \
				-e PLATFORM=test \
				$(_image_name):$(_test_image_version) $(_pytest)

_test_request_api := ./scripts/test_request_api.sh
_test_request_resnet50 := ./scripts/test_request_resnet50.sh

_train_iris := docker build \
				-t $(_train_iris_image_name):$(_train_iris_image_version) \
				-f $(_dockerfile_dir)/$(_train_iris_container_file) . && \
 			docker run --rm \
				--name $(_train_iris_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_train_iris_image_name):$(_train_iris_image_version)

_extract_resnet50 := docker build \
				-t $(_extract_resnet50_image_name):$(_extract_resnet50_image_version) \
				-f $(_dockerfile_dir)/$(_extract_resnet50_container_file) . && \
 			docker run --rm \
				--name $(_extract_resnet50_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_resnet50_image_name):$(_extract_resnet50_image_version)

_extract_inceptionv3 := docker build \
				-t $(_extract_inceptionv3_image_name):$(_extract_inceptionv3_image_version) \
				-f $(_dockerfile_dir)/$(_extract_inceptionv3_container_file) . && \
 			docker run --rm \
				--name $(_extract_inceptionv3_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_inceptionv3_image_name):$(_extract_inceptionv3_image_version)

.PHONY: pip_freeze
pip_freeze:
	$(_pip_freeze_test)

.PHONY: clean
clean:
	$(_compose_down)

.PHONY: c_build_api
c_build_api:
	$(_compose_build_api)

.PHONY: c_up_api
c_up_api:
	$(_compose_up_api)

.PHONY: c_down_api
c_down_api:
	$(_compose_down_api)

.PHONY: c_build_resnet50
c_build_resnet50:
	$(_compose_build_resnet50)

.PHONY: c_up_resnet50
c_up_resnet50:
	$(_compose_up_resnet50)

.PHONY: c_down_resnet50
c_down_resnet50:
	$(_compose_down_resnet50)

.PHONY: c_build_inceptionv3
c_build_inceptionv3:
	$(_compose_build_inceptionv3)

.PHONY: c_up_inceptionv3
c_up_inceptionv3:
	$(_compose_up_inceptionv3)

.PHONY: c_down_inceptionv3
c_down_inceptionv3:
	$(_compose_down_inceptionv3)

.PHONY: test_d
test_d:
	$(_pytest_d)

.PHONY: test_request_api
test_request_api:
	$(_test_request_api)

.PHONY: test_request_resnet50
test_request_resnet50:
	$(_test_request_resnet50)

.PHONY: train_iris
train_iris:
	$(_train_iris)

.PHONY: extract_resnet50
extract_resnet50:
	$(_extract_resnet50)

.PHONY: extract_inceptionv3
extract_inceptionv3:
	$(_extract_inceptionv3)
