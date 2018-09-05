#!/bin/sh

set -e

# https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup

# CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
CUDA_REPO_PKG=cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
dpkg -i /tmp/${CUDA_REPO_PKG}
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
rm -f /tmp/${CUDA_REPO_PKG}

# For libcudnn6
# docker run -it tensorflow/tensorflow:latest-gpu cat /etc/apt/sources.list.d/nvidia-ml.list
echo 'deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /' >/etc/apt/sources.list.d/nvidia-ml.list

apt update
apt install -y cuda-drivers cuda cuda-9-0 libcudnn7-dev python3-pip
pip3 install --upgrade pip tensorflow-gpu

# nvidia-cuda-toolkit

apt install -y nvidia-390 nvidia-docker2
