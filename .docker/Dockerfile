FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive=value

RUN apt update && apt install -y tzdata

RUN apt install --no-install-recommends -y \
    curl \
    wget \
    sudo \
    python3-tk \
    python3 \
    python3-pip

# キャッシュの削除
RUN apt-get autoremove -y && \
    apt clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --upgrade setuptools

# PyTorch
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY ./requirements.txt /tmp/

# 基本パッケージ
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt


# nodejsの導入
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
    && sudo apt-get install -y nodejs


#Jupyterlab 拡張
# 変数や行列の中身を確認
RUN jupyter labextension install @lckr/jupyterlab_variableinspector@3.0.7

# 自動整形
RUN pip install autopep8 \
    && pip install jupyterlab_code_formatter \
    && jupyter labextension install @ryantam626/jupyterlab_code_formatter \
    && jupyter serverextension enable --py jupyterlab_code_formatter

EXPOSE 8888

# CMD ["python3","-m","jupyter", "lab", "--ip=0.0.0.0", "--port=8888","--allow-root","--NotebookApp.token=''"]