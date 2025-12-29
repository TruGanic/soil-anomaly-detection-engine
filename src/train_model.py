import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# 1. Load the FINAL dataset
df = pd.read_csv('dataset/train_data_final.csv')

# 2. Select ALL Features (Including P, K, and Deltas)
features = [
    'Nitrogen', 'Phosphorus', 'Potassium', 'EC',
    'Delta_N', 'Delta_P', 'Delta_K', 'Delta_EC'
]

print(f"Training on {len(df)} samples with features: {features}")

# 3. Train Model
# contamination=0.01 means we assume 1% of real data might be noise
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(df[features])

# 4. Save
joblib.dump(model, 'models/isolation_forest_final.pkl')
print("✅ Model saved to models/isolation_forest_final.pkl")