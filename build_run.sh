docker build -t vlad/thesis -f Dockerfile .
docker run -ti \
    --volume $PWD:/workspace \
    vlad/thesis 