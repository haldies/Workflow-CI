FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yaml .
RUN conda env create -f conda.yaml

COPY . .

CMD ["/bin/bash", "-c", "source activate mlflow_env && mlflow run ."]