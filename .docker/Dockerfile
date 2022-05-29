FROM python:3.9.13-bullseye

RUN apt update \
    && apt install -y tk-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --upgrade setuptools

# jupyterLab
RUN pip install --no-cache-dir \
    jupyterlab 

# 基本パッケージ
RUN pip install --no-cache-dir \
    numpy \
    matplotlib 

