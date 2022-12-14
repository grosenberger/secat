# SECAT Dockerfile
FROM python:3.10.9

# Don't need this now that
# # Install R
# RUN apt update -qq
# RUN apt install -y --no-install-recommends software-properties-common dirmngr gnupg wget
# # add the signing key (by Michael Rutter) for these repos
# # To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# # Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
# RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 51716619E084DAB9
# RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
# RUN apt install -y --no-install-recommends r-base

# # Install BioConductor
# RUN Rscript -e 'install.packages("BiocManager")'

# # install VIPER
# RUN Rscript -e 'BiocManager::install(c("viper"))'

# # Install pip and carefully alias python3 as python
# RUN apt-get install -y python3-pip python-is-python3

# # install numpy
# RUN pip install numpy cython

# # install pyprophet
# RUN pip install pyprophet

# install SECAT and dependencies
ADD . /secat
WORKDIR /secat
RUN python setup.py install
WORKDIR /
RUN rm -rf /secat
