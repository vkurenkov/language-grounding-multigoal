FROM anibali/pytorch:cuda-9.0

# Hot-fi
USER root

# Requirements for OpenCV
RUN apt-get update
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

# Requirements for the repository
RUN pip install cython gym matplotlib tensorboard tensorboardX

WORKDIR /workspace