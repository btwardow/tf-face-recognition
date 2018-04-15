FROM tensorflow/tensorflow:1.7.0-py3

RUN apt-get update
RUN apt-get install -y git libfontconfig1 libxrender1 libsm6 libxext6 apt-utils
RUN apt-get clean

RUN pip --version
RUN pip install --upgrade pip

COPY docker/requirements.txt /server-requirements.txt
RUN pip install -r /server-requirements.txt

RUN mkdir /root/pretrained_models
COPY docker/download.py /root/pretrained_models/
COPY docker/download_vggace2.py /root/pretrained_models/
WORKDIR /root/pretrained_models
RUN python download.py
RUN python download_vggace2.py
RUN ls -l


COPY . /workspace

WORKDIR /workspace/
CMD cd server && ./server.sh