absolutePath := $(shell pwd)
imageName := sample_fastapi
imageVersion := latest
containerName := test
dockerBuild := docker build \
	-t $(imageName):$(imageVersion) .
dockerMount := -v $(absolutePath)/container_files/logs/gunicorn_error.log:/var/log/gunicorn_error.log \
	-v $(absolutePath)/container_files/logs/uvicorn_error.log:/var/log/uvicorn_error.log \
	-v $(absolutePath)/container_files/logs/gunicorn_access.log:/var/log/gunicorn_access.log \
	-v $(absolutePath)/container_files/logs/uvicorn_access.log:/var/log/uvicorn_access.log \
	-v $(absolutePath)/container_files/data/:/fastapi/app/data/
dockerParams := -it -d -p 8888:8888 --name $(containerName)
dockerRunGunicorn := docker run \
	$(dockerParams) \
	-e GU=G \
	$(dockerMount) \
	$(imageName):$(imageVersion)
dockerRunUvicorn := docker run \
	$(containerName) \
	-e GU=U \
	$(dockerMount) \
	$(imageName):$(imageVersion)
dockerRerun := docker restart $(containerName)
requirements := requirements.txt
pipFreeze := pip freeze > $(requirements)

.PHONY: build run build_run rerun clean pip_freeze
pip_freeze:
	$(pipFreeze)

build: pip_freeze
	$(dockerBuild)

run:
	$(dockerRunGunicorn)

runU:
	$(dockerRunUvicorn)

build_run: build run

rerun:
	$(dockerRerun)

clean:
	$(dockerRm)
	docker rmi -f $(imageName):$(imageVersion)

