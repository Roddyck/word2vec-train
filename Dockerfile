FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app

COPY . /app

RUN python3 -m pip install -r requirements.txt
