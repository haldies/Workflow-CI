# Gunakan image dasar dengan Miniconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Salin isi project
COPY . /app

# Salin environment file dan install
RUN conda env create -f conda.yaml

# Aktifkan env dan jalankan script
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

CMD ["python", "modelling.py"]
