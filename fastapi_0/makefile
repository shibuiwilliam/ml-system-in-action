imageName := sample_fastapi
imageVersion := latest
containerName := test
dockerBuild := docker build -t $(imageName):$(imageVersion) .
dockerRun := docker run -it -d -p 8888:8888 --name $(containerName) $(imageName):$(imageVersion)
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
	$(dockerRun)

build_run: build run

rerun:
	$(dockerRerun)
	$(dockerRun)

clean:
	$(dockerRm)
	docker rmi -f $(imageName):$(imageVersion)

