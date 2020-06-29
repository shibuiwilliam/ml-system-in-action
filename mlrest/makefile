absolutePath := $(shell pwd)
imageName := mlrest_app
imageVersion := latest
containerName := mlrest_app

composeBuild := docker-compose build
composeUp := docker-compose up -d
composeDown := docker-compose down

requirements := requirements.txt
pipFreeze := pip freeze > $(requirements)

.PHONY: clean pip_freeze dirs compose_build compose_up compose_down
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
