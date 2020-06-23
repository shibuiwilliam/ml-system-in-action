imageName := sample_fastapi
imageVersion := latest
containerName := test
dockerBuild := docker build -t $(imageName):$(imageVersion) .
dockerRunGunicorn := docker run -it -d -p 8888:8888 -e GU=G --name $(containerName) $(imageName):$(imageVersion)
dockerRunUvicorn := docker run -it -d -p 8888:8888 -e GU=U --name $(containerName) $(imageName):$(imageVersion)
dockerRm := docker ps -q --filter "name=$(containerName)" | grep -q . && docker rm -f $(containerName)
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
	$(dockerRun)

clean:
	$(dockerRm)
	docker rmi -f $(imageName):$(imageVersion)

