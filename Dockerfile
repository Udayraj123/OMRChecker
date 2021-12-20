FROM python:3.8.7-slim-buster

WORKDIR /app

# Use bash instead of sh
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Enable non-interactively
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

## Update Distro
RUN apt update -y \
    && apt-get update -y \
    && apt install curl sudo -y

## Install Dependencies
RUN apt-get install -y build-essential cmake unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libatlas-base-dev gfortran

## Setup Python
RUN pip3 install --user opencv-python opencv-contrib-python
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

## Clean up
RUN apt-get clean
RUN pip3 cache purge
RUN rm -rf /var/lib/apt/lists/*

## Setup environment
ENTRYPOINT /bin/bash
