import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# 1. Load Data
df = pd.read_csv('dataset/train_data.csv')
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'EC']

# 2. Initialize Model
# contamination=0.01 means we expect ~1% of our "organic" training data might be noise.
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# 3. Train on "Normal" Organic Data only
print("Training model on organic patterns...")
model.fit(df[features])

# 4. Save the Model
joblib.dump(model, 'models/isolation_forest_soil.pkl')
print("Model saved to models/isolation_forest_soil.pkl")

# --- Simple Test ---
# -1 means Anomaly, 1 means Normal
test_sample = [[120, 45, 160, 3.5]] # High N and EC (Chemical)
prediction = model.predict(test_sample)
print(f"Test Prediction (1=Organic, -1=Chemical): {prediction[0]}")