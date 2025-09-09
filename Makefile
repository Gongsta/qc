UID := $(shell id -u)
GID := $(shell id -g)

# To set up docker first time
# sudo usermod -aG docker $USER
# pkill -u ubuntu
# docker login

build-image-slurm:
	srun --nodelist=trpro-slurm2 --cpus-per-task=19 --gres tmpdisk:40960 bash -lc 'hostname; DOCKER_DATA_ROOT=/home/s36gong/steven-docker-images slurm-start-dockerd.sh; \
	    docker build -t qc .'

build-image:
	docker build -t qc .

push-image:
	docker tag qc docker.io/gongsta/qc:latest
	docker push docker.io/gongsta/qc:latest

import-enroot:
	enroot import -o qc.sqsh dockerd://gongsta/qc:latest
	enroot create qc.sqsh

start-enroot:
	enroot start --rw --conf enroot-config/conf qc

start-enroot-root:
	enroot start qc


start-container:
	docker run -it --gpus all --rm --name qc \
	  -v $(shell pwd):/app/qc -w /app/qc \
	  -v /etc/passwd:/etc/passwd:ro \
	  -v /etc/group:/etc/group:ro \
	  gongsta/qc:latest

# start-container:
# 	docker run -it --gpus all --rm --name qc \
# 	  --user "$(UID):$(GID)" \
# 	  -v $(shell pwd):/app/qc -w /app/qc \
# 	  -v /etc/passwd:/etc/passwd:ro \
# 	  -v /etc/group:/etc/group:ro \
# 	  -e USER=$(shell id -un) \
# 	  qc

start-container-slurm:
	srun --nodelist=trpro-slurm2 --cpus-per-task=9 --gres tmpdisk:40960,shard:8192 --pty bash -lc 'hostname; DOCKER_DATA_ROOT=/home/s36gong/steven-docker-images slurm-start-dockerd.sh; \
	    docker run -it --gpus all --rm --name qc -v /home/s36gong/qc:/app/qc qc'
