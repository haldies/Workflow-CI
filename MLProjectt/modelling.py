import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
import joblib

# mlflow.set_tracking_uri("http://localhost:8000")

# Set experiment (jika belum ada, akan dibuat)
mlflow.set_experiment("iris-logistic-regression")

mlflow.sklearn.autolog()

X_train = joblib.load("../Workflow-CI/iris_preproceccing/X_train.pkl")
X_test = joblib.load("../Workflow-CI/iris_preproceccing/X_test.pkl")
y_train = joblib.load("../Workflow-CI/iris_preproceccing/y_train.pkl")
y_test = joblib.load("../Workflow-CI/iris_preproceccing/y_test.pkl")

with mlflow.start_run():
    model = LogisticRegression(C=0.5, solver='liblinear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)

    print("Model dilatih dan dicatat secara otomatis di MLflow dengan metrik.")

