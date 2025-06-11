import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Aktifkan autologging (MLflow akan otomatis memulai run saat digunakan via `mlflow run`)
mlflow.sklearn.autolog()

# Path ke file preprocessing
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "titanic_preprocessing", "titanic_preprocessed_train.csv")

# Baca dataset
df = pd.read_csv(csv_path)

# Drop kolom yang tidak dibutuhkan
df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId", "Ticket_number", "Ticket_item"], errors='ignore')

# Encode kolom kategorikal
label_cols = df.select_dtypes(include='object').columns
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Pisahkan fitur dan target
X = df.drop(columns=["Survived"])
y = df["Survived"]

print("Fitur:", X.columns.tolist())
print("Jumlah fitur:", X.shape[1])

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Logging manual (opsional karena autolog sudah mencatat ini juga)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)


print(f"✅ Model dilatih dan dicatat di MLflow.")
print(f"🔍 Akurasi: {accuracy:.4f}")
