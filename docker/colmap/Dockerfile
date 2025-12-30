FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libsqlite3-dev \
    libopencv-dev \
    libcgal-dev \
    libceres-dev \
    libflann-dev \
    libmetis-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    git checkout 3.8 && \
    mkdir build && cd build && \
    cmake .. -DCUDA_ENABLED=OFF && \
    make -j$(nproc) && \
    make install

WORKDIR /workspace

CMD ["colmap", "--help"]
