# SECAT Dockerfile
FROM python:3.7.3

# install R
RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base

# install VIPER
RUN Rscript -e 'source("http://bioconductor.org/biocLite.R")' -e 'biocLite("viper")'

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
