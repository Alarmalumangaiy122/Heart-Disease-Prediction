import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("heart.xlsx")

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save Columns
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Heart Disease Model saved successfully!")
