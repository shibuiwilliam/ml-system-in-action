_absolute_path := $(shell pwd)
_dockerfile_dir := dockerfiles
_dockercompose_dir := docker-composes

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

_extract_resnet50_onnx_container_file := Dockerfile_extract_resnet50_onnx
_extract_resnet50_onnx_image_name := serving_patterns_extract_resnet50_onnx
_extract_resnet50_onnx_image_version := latest
_extract_resnet50_onnx_container_name := serving_patterns_extract_resnet50_onnx

_extract_resnet50_tfs_container_file := Dockerfile_extract_resnet50_tfs
_extract_resnet50_tfs_image_name := serving_patterns_extract_resnet50_tfs
_extract_resnet50_tfs_image_version := latest
_extract_resnet50_tfs_container_name := serving_patterns_extract_resnet50_tfs

_extract_inceptionv3_container_file := Dockerfile_extract_inceptionv3
_extract_inceptionv3_image_name := serving_patterns_extract_inceptionv3
_extract_inceptionv3_image_version := latest
_extract_inceptionv3_container_name := serving_patterns_extract_inceptionv3

_extract_mobilenetv2_container_file := Dockerfile_extract_mobilenetv2
_extract_mobilenetv2_image_name := serving_patterns_extract_mobilenetv2
_extract_mobilenetv2_image_version := latest
_extract_mobilenetv2_container_name := serving_patterns_extract_mobilenetv2

_inceptionv3_tfs_container_file := Dockerfile_tfserving_inceptionv3
_inceptionv3_tfs_image_name := serving_patterns_inceptionv3_tfs
_inceptionv3_tfs_image_version := latest
_inceptionv3_tfs_container_name := serving_patterns_inceptionv3_tfs

_inceptionv3_api_container_file := Dockerfile_api_inceptionv3
_inceptionv3_api_image_name := serving_patterns_inceptionv3_api
_inceptionv3_api_image_version := latest
_inceptionv3_api_container_name := serving_patterns_inceptionv3_api

_mobilenetv2_tfs_container_file := Dockerfile_tfserving_mobilenetv2
_mobilenetv2_tfs_image_name := serving_patterns_mobilenetv2_tfs
_mobilenetv2_tfs_image_version := latest
_mobilenetv2_tfs_container_name := serving_patterns_mobilenetv2_tfs

_mobilenetv2_api_container_file := Dockerfile_api_mobilenetv2
_mobilenetv2_api_image_name := serving_patterns_mobilenetv2_api
_mobilenetv2_api_image_version := latest
_mobilenetv2_api_container_name := serving_patterns_mobilenetv2_api

_compose_api := $(_dockercompose_dir)/docker-compose_api.yml
_compose_build_api := docker-compose -f $(_compose_api) build
_compose_up_api := docker-compose -f $(_compose_api) up -d
_compose_down_api := docker-compose -f $(_compose_api) down

_compose_resnet50_onnx := $(_dockercompose_dir)/docker-compose_resnet50_onnx.yml
_compose_build_resnet50_onnx := docker-compose -f $(_compose_resnet50_onnx) build
_compose_up_resnet50_onnx := docker-compose -f $(_compose_resnet50_onnx) up -d
_compose_down_resnet50_onnx := docker-compose -f $(_compose_resnet50_onnx) down

_compose_resnet50_tfs := $(_dockercompose_dir)/docker-compose_resnet50_tfs.yml
_compose_build_resnet50_tfs := docker-compose -f $(_compose_resnet50_tfs) build
_compose_up_resnet50_tfs := docker-compose -f $(_compose_resnet50_tfs) up -d
_compose_down_resnet50_tfs := docker-compose -f $(_compose_resnet50_tfs) down

_compose_inceptionv3 := $(_dockercompose_dir)/docker-compose_inceptionv3.yml
_compose_build_inceptionv3 := docker-compose -f $(_compose_inceptionv3) build
_compose_up_inceptionv3 := docker-compose -f $(_compose_inceptionv3) up -d
_compose_down_inceptionv3 := docker-compose -f $(_compose_inceptionv3) down

_compose_mobilenetv2 := $(_dockercompose_dir)/docker-compose_mobilenetv2.yml
_compose_build_mobilenetv2 := docker-compose -f $(_compose_mobilenetv2) build
_compose_up_mobilenetv2 := docker-compose -f $(_compose_mobilenetv2) up -d
_compose_down_mobilenetv2 := docker-compose -f $(_compose_mobilenetv2) down

_compose_api_ms_horizontal := $(_dockercompose_dir)/docker-compose_api_ms_horizontal.yml
_compose_build_api_ms_horizontal := docker-compose -f $(_compose_api_ms_horizontal) build
_compose_up_api_ms_horizontal := docker-compose -f $(_compose_api_ms_horizontal) up -d
_compose_down_api_ms_horizontal := docker-compose -f $(_compose_api_ms_horizontal) down

_compose_image_ms_horizontal := $(_dockercompose_dir)/docker-compose_image_ms_horizontal.yml
_compose_build_image_ms_horizontal := docker-compose -f $(_compose_image_ms_horizontal) build
_compose_up_image_ms_horizontal := docker-compose -f $(_compose_image_ms_horizontal) up -d
_compose_down_image_ms_horizontal := docker-compose -f $(_compose_image_ms_horizontal) down

_requirements_test := requirements_test.txt
_pip_freeze_test := pip freeze > requirements/$(_requirements_test)
_pip_install_test := pip install -r requirements/$(_requirements_test)

_test_request_api := ./scripts/test_request_api.sh
_test_request_resnet50_onnx := ./scripts/test_request_resnet50_onnx.sh
_test_request_inceptionv3 := ./scripts/test_request_inceptionv3.sh
_test_request_mobilenetv2 := ./scripts/test_request_mobilenetv2.sh
_test_request_image_ms_horizontal := ./scripts/test_request_image_ms_horizontal.sh

_pytest_d := docker build \
				-t $(_image_name):$(_test_image_version) \
				-f $(_dockerfile_dir)/$(_test_container_file) . && \
			docker run --rm \
				--name $(_test_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				-e PROFILE=0 \
				-e PLATFORM=test \
				$(_image_name):$(_test_image_version) pytest -v -s ./tests/

_train_iris := docker build \
				-t $(_train_iris_image_name):$(_train_iris_image_version) \
				-f $(_dockerfile_dir)/$(_train_iris_container_file) . && \
 			docker run --rm \
				--name $(_train_iris_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_train_iris_image_name):$(_train_iris_image_version)

_extract_resnet50_onnx := docker build \
				-t $(_extract_resnet50_onnx_image_name):$(_extract_resnet50_onnx_image_version) \
				-f $(_dockerfile_dir)/$(_extract_resnet50_onnx_container_file) . && \
 			docker run --rm \
				--name $(_extract_resnet50_onnx_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_resnet50_onnx_image_name):$(_extract_resnet50_onnx_image_version)

_extract_resnet50_tfs := docker build \
				-t $(_extract_resnet50_tfs_image_name):$(_extract_resnet50_tfs_image_version) \
				-f $(_dockerfile_dir)/$(_extract_resnet50_tfs_container_file) . && \
 			docker run --rm \
				--name $(_extract_resnet50_tfs_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_resnet50_tfs_image_name):$(_extract_resnet50_tfs_image_version)

_extract_inceptionv3 := docker build \
				-t $(_extract_inceptionv3_image_name):$(_extract_inceptionv3_image_version) \
				-f $(_dockerfile_dir)/$(_extract_inceptionv3_container_file) . && \
 			docker run --rm \
				--name $(_extract_inceptionv3_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_inceptionv3_image_name):$(_extract_inceptionv3_image_version)

_extract_mobilenetv2 := docker build \
				-t $(_extract_mobilenetv2_image_name):$(_extract_mobilenetv2_image_version) \
				-f $(_dockerfile_dir)/$(_extract_mobilenetv2_container_file) . && \
 			docker run --rm \
				--name $(_extract_mobilenetv2_container_name) \
				-v $(_absolute_path)/models/:/serving_patterns/models/ \
				$(_extract_mobilenetv2_image_name):$(_extract_mobilenetv2_image_version)

_build_locust := docker build -t locust:latest -f dockerfiles/Dockerfile_locust .

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

.PHONY: c_build_resnet50_onnx
c_build_resnet50_onnx:
	$(_compose_build_resnet50_onnx)

.PHONY: c_up_resnet50_onnx
c_up_resnet50_onnx:
	$(_compose_up_resnet50_onnx)

.PHONY: c_down_resnet50_onnx
c_down_resnet50_onnx:
	$(_compose_down_resnet50_onnx)

.PHONY: c_build_resnet50_tfs
c_build_resnet50_tfs:
	$(_compose_build_resnet50_tfs)

.PHONY: c_up_resnet50_tfs
c_up_resnet50_tfs:
	$(_compose_up_resnet50_tfs)

.PHONY: c_down_resnet50_tfs
c_down_resnet50_tfs:
	$(_compose_down_resnet50_tfs)

.PHONY: c_build_inceptionv3
c_build_inceptionv3:
	$(_compose_build_inceptionv3)

.PHONY: c_up_inceptionv3
c_up_inceptionv3:
	$(_compose_up_inceptionv3)

.PHONY: c_down_inceptionv3
c_down_inceptionv3:
	$(_compose_down_inceptionv3)

.PHONY: c_build_mobilenetv2
c_build_mobilenetv2:
	$(_compose_build_mobilenetv2)

.PHONY: c_up_mobilenetv2
c_up_mobilenetv2:
	$(_compose_up_mobilenetv2)

.PHONY: c_down_mobilenetv2
c_down_mobilenetv2:
	$(_compose_down_mobilenetv2)

.PHONY: c_build_api_ms_h
c_build_api_ms_h:
	$(_compose_build_api_ms_horizontal)

.PHONY: c_up_api_ms_h
c_up_api_ms_h:
	$(_compose_up_api_ms_horizontal)

.PHONY: c_down_api_ms_h
c_down_api_ms_h:
	$(_compose_down_api_ms_horizontal)

.PHONY: c_build_image_ms_h
c_build_image_ms_h:
	$(_compose_build_image_ms_horizontal)

.PHONY: c_up_image_ms_h
c_up_image_ms_h:
	$(_compose_up_image_ms_horizontal)

.PHONY: c_down_image_ms_h
c_down_image_ms_h:
	$(_compose_down_image_ms_horizontal)

.PHONY: test_d
test_d:
	$(_pytest_d)

.PHONY: test_request_api
test_request_api:
	$(_test_request_api)

.PHONY: test_request_resnet50_onnx
test_request_resnet50_onnx:
	$(_test_request_resnet50_onnx)

.PHONY: test_request_inceptionv3
test_request_inceptionv3:
	$(_test_request_inceptionv3)

.PHONY: test_request_mobilenetv2
test_request_mobilenetv2:
	$(_test_request_mobilenetv2)

.PHONY: test_request_image_ms_horizontal
test_request_image_ms_horizontal:
	$(_test_request_image_ms_horizontal)

.PHONY: build_locust
build_locust:
	$(_build_locust)

.PHONY: run_locust
run_locust:
	docker run -it --rm -p 8089:8089 --name locust locust:latest locust -f ${ARGS}

.PHONY: train_iris
train_iris:
	$(_train_iris)

.PHONY: extract_resnet50_onnx
extract_resnet50_onnx:
	$(_extract_resnet50_onnx)

.PHONY: extract_resnet50_tfs
extract_resnet50_tfs:
	$(_extract_resnet50_tfs)

.PHONY: extract_inceptionv3
extract_inceptionv3:
	$(_extract_inceptionv3)

.PHONY: extract_mobilenetv2
extract_mobilenetv2:
	$(_extract_mobilenetv2)

.PHONY: extract_all
extract_all: train_iris extract_resnet50_onnx extract_resnet50_tfs extract_inceptionv3 extract_mobilenetv2 test_d

.PHONY: build_all
build_all: extract_all c_build_api c_build_resnet50_onnx c_build_resnet50_tfs c_build_inceptionv3 c_build_mobilenetv2 build_locust
