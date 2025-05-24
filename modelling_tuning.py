import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score

# Load data
X_train = joblib.load("iris_preprocessing/X_train.pkl")
X_test = joblib.load("iris_preprocessing/X_test.pkl")
y_train = joblib.load("iris_preprocessing/y_train.pkl")
y_test = joblib.load("iris_preprocessing/y_test.pkl")

# Grid search
param_grid = {
    'C': [0.1, 0.5, 1.0],
    'solver': ['liblinear', 'lbfgs']
}

# Tracking
for C in param_grid['C']:
    for solver in param_grid['solver']:
        with mlflow.start_run():
            model = LogisticRegression(C=C, solver=solver, max_iter=200)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')

            
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)

            mlflow.log_metric("num_iter", model.n_iter_[0])
            mlflow.log_metric("train_score", model.score(X_train, y_train))

            # Save model
            mlflow.sklearn.log_model(model, "model")

print("Tuning selesai. Cek MLflow UI untuk detail.")

