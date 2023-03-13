FROM ubuntu:18.04
LABEL maintainer="yahwang"

ARG PYTHON_VERSION=3.8.16
ARG BUILD_PYTHON_DEPS=" \
        make \
        build-essential \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libncurses5-dev \
        libncursesw5-dev \ 
        libnss3-dev \
        libreadline-dev \
        libsqlite3-dev \ 
        libssl-dev \
        liblzma-dev \
        xz-utils \ 
        zlib1g-dev \
        "
ARG BUILD_OPT_DEPS=" \
        sudo \
        locales \
        git \
        ca-certificates \
        curl\
        megatools \
        "

# basic update & locale setting
RUN apt-get update \
 && apt-get upgrade -yqq \
 && apt-get install -y --no-install-recommends \
        ${BUILD_PYTHON_DEPS} \
        ${BUILD_OPT_DEPS} \
 && localedef -f UTF-8 -i en_US en_US.UTF-8 \
 && useradd -m -s /bin/bash ubuntu \
 && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && apt-get autoremove -yqq --purge \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER ubuntu

WORKDIR /home/ubuntu

ENV LANG=en_US.UTF-8 \
    PYENV_ROOT="/home/ubuntu/.pyenv" \
    PATH="/home/ubuntu/.pyenv/bin:/home/ubuntu/.pyenv/shims:$PATH"

# install pyenv & python
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash \
 && pyenv install ${PYTHON_VERSION} \
 && pyenv global ${PYTHON_VERSION} \
 && pip install --upgrade pip

RUN git clone https://github.com/alessandrocuda/ReSt
WORKDIR /home/ubuntu/ReSt/results/model/word2vec
RUN megadl 'https://mega.nz/#!QpsB2JrK!Tnv-YnSvGVPuoZDFt4jfKiDpyXdwjvAJJgmmKcV-vZI'
WORKDIR /home/ubuntu/ReSt/
RUN pip install -Ur requirements.txt
WORKDIR /home/ubuntu/ReSt/app

CMD ["flask", "run", "--host", "0.0.0.0", "-p","3000" ]
#RUN flask run