import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = load_iris(as_frame=True)
df = data.frame

print("Piersze 5 wierszy:")
print(df.head())
print("\nRozmiar danych:", df.shape)
print("\nTypy kolumn:")
print(df.dtypes)

print("=================== Zadanie 2 =======================")

X = df.drop(columns=["target"])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred)}")
print(classification_report(y_test, pred))


print("=================== Zadanie 3 =======================")

joblib.dump(model, 'models/model_v1.joblib')
print("Model saved as 'models/model_v1.joblib'")