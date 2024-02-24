FROM python:3.11-slim

WORKDIR /app

# Enable non-interactively
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

## Update Distro
RUN apt-get update -y

## Install Dependencies
RUN apt-get install -y build-essential cmake unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libatlas-base-dev gfortran

#Copy files over
COPY ./ /app

## Setup Python
RUN pip3 install --user opencv-python opencv-contrib-python
RUN pip3 install -r requirements.txt

## Clean up
RUN apt-get clean
RUN apt-get -y autoremove
RUN pip3 cache purge

#Set envivronment variables
ENV OMR_CHECKER_CONTAINER True

## Run the default command
CMD ["python3", "/app/main.py", "--inputDir", "/app/inputs/", "--outputDir", "/app/outputs/"]
#^^^ ensure that inputs and outputs are in the appropriate location.
# If changes are made to the program, they should be reflected here and in
# the instructional materials.
