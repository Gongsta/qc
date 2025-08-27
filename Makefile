build-image-slurm:
	srun --nodelist=trpro-slurm2 bash -lc 'hostname; slurm-start-dockerd.sh; \
	    docker build -t qc .'

build-image:
	docker build -t qc .
