# SECAT Dockerfile
FROM python:3.10.9

# install numpy
RUN pip install numpy cython

# install pyprophet
RUN pip install pyprophet

# install SECAT and dependencies
ADD . /secat
WORKDIR /secat
RUN python setup.py install
WORKDIR /
RUN rm -rf /secat
