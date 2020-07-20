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

_extract_imagenet_resnet50_container_file := Dockerfile_extract_imagenet_resnet50
_extract_imagenet_resnet50_image_name := serving_patterns_extract_imagenet_resnet50
_extract_imagenet_resnet50_image_version := latest
_extract_imagenet_resnet50_container_name := serving_patterns_extract_imagenet_resnet50

_extract_imagenet_inceptionv3_container_file := Dockerfile_extract_imagenet_inceptionv3
_extract_imagenet_inceptionv3_image_name := serving_patterns_extract_imagenet_inceptionv3
_extract_imagenet_inceptionv3_image_version := latest
_extract_imagenet_inceptionv3_container_name := serving_patterns_extract_imagenet_inceptionv3

_imagenet_inceptionv3_tfs_container_file := Dockerfile_tfserving_inceptionv3
_imagenet_inceptionv3_tfs_image_name := serving_patterns_imagenet_inceptionv3_tfs
_imagenet_inceptionv3_tfs_image_version := latest
_imagenet_inceptionv3_tfs_container_name := serving_patterns_imagenet_inceptionv3_tfs

_imagenet_inceptionv3_front_container_file := Dockerfile_front_inceptionv3
_imagenet_inceptionv3_front_image_name := serving_patterns_imagenet_inceptionv3_front
_imagenet_inceptionv3_front_image_version := latest
_imagenet_inceptionv3_front_container_name := serving_patterns_imagenet_inceptionv3_front

_compose_build := docker-compose build
_compose_up := docker-compose up -d
_compose_down := docker-compose down

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
				$(_image_name):$(_test_image_version) $(_pytest)

_test_request := ./scripts/test_request.sh

_train_iris := docker build \
				-t $(_train_iris_image_name):$(_train_iris_image_version) \
				-f $(_dockerfile_dir)/$(_train_iris_container_file) . && \
 			docker run --rm \
				--name $(_train_iris_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_train_iris_image_name):$(_train_iris_image_version)

_extract_imagenet_resnet50 := docker build \
				-t $(_extract_imagenet_resnet50_image_name):$(_extract_imagenet_resnet50_image_version) \
				-f $(_dockerfile_dir)/$(_extract_imagenet_resnet50_container_file) . && \
 			docker run --rm \
				--name $(_extract_imagenet_resnet50_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_imagenet_resnet50_image_name):$(_extract_imagenet_resnet50_image_version)

_extract_imagenet_inceptionv3 := docker build \
				-t $(_extract_imagenet_inceptionv3_image_name):$(_extract_imagenet_inceptionv3_image_version) \
				-f $(_dockerfile_dir)/$(_extract_imagenet_inceptionv3_container_file) . && \
 			docker run --rm \
				--name $(_extract_imagenet_inceptionv3_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_imagenet_inceptionv3_image_name):$(_extract_imagenet_inceptionv3_image_version)

_run_inceptionv3_tfserving := docker build \
				-t $(_imagenet_inceptionv3_tfs_image_name):$(_imagenet_inceptionv3_tfs_image_version) \
				-f $(_dockerfile_dir)/$(_imagenet_inceptionv3_tfs_container_file) . && \
			docker run --rm \
				--name $(_imagenet_inceptionv3_tfs_container_name) \
				-p 8500:8500 \
				-p 8501:8501 \
				$(_imagenet_inceptionv3_tfs_image_name):$(_imagenet_inceptionv3_tfs_image_version)

.PHONY: clean pip_freeze compose_build compose_up compose_down test test_d train_iris test_request extract_imagenet_resnet50 extract_imagenet_inceptionv3 run_inceptionv3_tfserving
pip_freeze:
	$(_pip_freeze_test)

clean:
	$(_compose_down)

compose_build:
	$(_compose_build)

compose_up:
	$(_compose_up)

compose_down:
	$(_compose_down)

test:
	$(_pytest)

test_d:
	$(_pytest_d)

test_request:
	$(_test_request)

train_iris:
	$(_train_iris)

extract_imagenet_resnet50:
	$(_extract_imagenet_resnet50)

extract_imagenet_inceptionv3:
	$(_extract_imagenet_inceptionv3)

run_inceptionv3_tfserving:
	$(_run_inceptionv3_tfserving)