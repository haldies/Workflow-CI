import os
from prometheus_client import start_http_server, Gauge
import time

def get_latest_metric():
    mlruns_path = './mlruns/233895560391994134/5a4eccad7831467fa0f4a8b6afc0bae4/metrics/accuracy'  # sesuaikan path
    if not os.path.exists(mlruns_path):
        return None
    with open(mlruns_path, 'r') as f:
        data = f.readlines()
        if not data:
            return None
        last_line = data[-1].strip()
        parts = last_line.split()
        if len(parts) >= 2:
            _, value = parts[0], parts[1]
            return float(value)
        else:
            return None

if __name__ == '__main__':
    accuracy_gauge = Gauge('mlflow_training_accuracy', 'Latest training accuracy from MLflow')

    start_http_server(8000)
    print("Serving metrics on port 8000")

    while True:
        accuracy = get_latest_metric()
        if accuracy is not None:
            accuracy_gauge.set(accuracy)
        time.sleep(10)
