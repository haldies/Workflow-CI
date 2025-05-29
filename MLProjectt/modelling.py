import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ci-training")

base_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(base_dir, "..", "preprocessing",
                        "titanic_preprocessed_train.csv")

df = pd.read_csv(csv_path)


mlflow.sklearn.autolog()
df = df.drop(columns=["Name", "Ticket", "Cabin",
             "PassengerId"], errors='ignore')

label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print(f"✅ Model dilatih dan dicatat di MLflow.")
    print(f"🔍 Akurasi: {accuracy:.4f}")
