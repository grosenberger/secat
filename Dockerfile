# SECAT Dockerfile
FROM python:3.10.8

# install numpy
RUN pip install numpy==1.23.5 cython

# install pyprophet
RUN pip install pyprophet

# install SECAT and dependencies
ADD . /secat
WORKDIR /secat
RUN python setup.py install
WORKDIR /
RUN rm -rf /secat
