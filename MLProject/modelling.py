import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "titanic_preprocessing", "titanic_preprocessed_train.csv")

df = pd.read_csv(csv_path)

mlflow.set_experiment("titanic_experiment")



df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId", "Ticket_number", "Ticket_item"], errors='ignore')

label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]

print("Fitur:", X.columns.tolist())
print("Jumlah fitur:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    signature = infer_signature(X_train, model.predict(X_train))

    
    input_example = X_train.head(5)

    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print(f"Akurasi: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    print("Run ID:", run.info.run_id)