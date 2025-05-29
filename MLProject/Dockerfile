FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yaml .
RUN conda env create -f conda.yaml

COPY . .

CMD ["conda", "run", "-n", "mlflow_env", "mlflow", "run", "."]