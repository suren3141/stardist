FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

ARG NVIDIA_DRIVER_VERSION=470

RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    libnvidia-compute-${NVIDIA_DRIVER_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip
RUN pip install stardist gputools edt opencv-python3