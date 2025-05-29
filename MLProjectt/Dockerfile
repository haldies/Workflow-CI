FROM python:3.8-slim

WORKDIR /app

COPY conda.yaml ./
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda env create -f conda.yaml && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/mlflow_env/bin:$PATH

COPY . .

CMD ["mlflow", "run", "."]
