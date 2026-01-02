import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("gesture_data.csv", header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved")
