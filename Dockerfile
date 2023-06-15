FROM ubuntu

WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y libopenmpi-dev && \
    apt-get install -y python3-pip && \
    git clone https://github.com/kitamoto-lab/pyphoon2.git && \
    cd pyphoon2 &&  \
    pip3 install . && \
    pip3 install tqdm && \
    pip3 install scikit-learn && \
    pip3 install matplotlib && \
    pip3 install seaborn && \
    pip3 install lightning && \
    pip3 install tensorboardX

