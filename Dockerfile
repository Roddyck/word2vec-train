FROM nvidia/cuda:12.1.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

RUN python3 -m pip install torch torchvision torchaudio

COPY . /app
