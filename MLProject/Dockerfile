FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the conda environment file
COPY conda.yaml .

# Install necessary tools and Miniconda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda env create -f conda.yaml && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/mlflow_env/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=mlflow_env

COPY . .

CMD ["conda", "run", "-n", "mlflow_env", "mlflow", "run", "."]
