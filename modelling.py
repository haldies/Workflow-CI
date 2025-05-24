import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
import joblib

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load data
X_train = joblib.load("../MLProject/iris_preproceccing/X_train.pkl")
X_test = joblib.load("../MLProject/iris_preproceccing/X_test.pkl")
y_train = joblib.load("../MLProject/iris_preproceccing/y_train.pkl")
y_test = joblib.load("../MLProject/iris_preproceccing/y_test.pkl")

with mlflow.start_run():
    # Model
    model = LogisticRegression(C=0.5, solver='liblinear')
    model.fit(X_train, y_train)

    # Predict (opsional jika ingin cetak/cek manual)
    y_pred = model.predict(X_test)

    print("Model dilatih dan dicatat secara otomatis di MLflow.")
