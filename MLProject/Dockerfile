FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

RUN python modelling.py

CMD ["conda", "run", "-n", "mlflow-env", "uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
