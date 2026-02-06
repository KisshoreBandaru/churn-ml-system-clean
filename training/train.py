from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# Create simple fake data (for learning)
data = pd.DataFrame({
    "tenure": [1, 5, 10, 20, 30, 40],
    "monthly_charges": [50, 60, 70, 80, 90, 100],
    "total_charges": [50, 300, 700, 1600, 2700, 4000],
    "support_calls": [5, 4, 3, 1, 0, 0],
    "contract_type": ["monthly", "monthly", "yearly", "yearly", "yearly", "yearly"],
    "churn": [1, 1, 0, 0, 0, 0]
})
data = pd.get_dummies(data, columns=["contract_type"])

X = data.drop("churn", axis=1)
y = data["churn"]

model = LogisticRegression()
model.fit(X, y)

# SAVE MODEL
joblib.dump(model, "models/churn_v3.pkl")

print("✅ Model saved as models/churn_v3.pkl")
