docker build -t vlad/thesis -f Dockerfile .
docker run --runtime=nvidia -ti \
    --volume $PWD:/workspace \
    vlad/thesis /bin/bash