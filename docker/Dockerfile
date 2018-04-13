FROM tensorflow/tensorflow:1.7.0-py3

RUN apt-get update
RUN apt-get install -y git libfontconfig1 libxrender1 libsm6 libxext6 apt-utils
RUN apt-get clean

RUN pip --version
RUN pip install --upgrade pip

COPY requirements.txt /server-requirements.txt
RUN pip install -r /server-requirements.txt

WORKDIR /workspace/server/
CMD ./server.sh
