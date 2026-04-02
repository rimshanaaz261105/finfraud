import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create a tiny fake dataset to "train" a model for the demo
data = pd.DataFrame({
    'amount': [10, 5000, 20, 8000],
    'oldbalanceOrg': [100, 5000, 200, 100],
    'merchant': [1, 2, 1, 3],
    'isFraud': [0, 1, 0, 1] # 0 = Safe, 1 = Fraud
})

X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save as model.pkl
joblib.dump(model, 'model.pkl')
print("model.pkl has been created successfully!")
