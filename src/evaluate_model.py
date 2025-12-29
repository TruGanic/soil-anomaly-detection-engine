import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate():
    print("--- 📊 Loading Test Data & Model ---")
    
    # 1. Load the Test Data (Mixed Organic & Chemical)
    try:
        df_test = pd.read_csv('dataset/test_data_final.csv')
        print(f"✅ Loaded {len(df_test)} test samples.")
    except FileNotFoundError:
        print("❌ Error: 'dataset/test_data_final.csv' not found.")
        return

    # 2. Load the Trained Model
    try:
        model = joblib.load('models/isolation_forest_final.pkl')
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Model file not found. Please run train_model.py first.")
        return

    # 3. Define Features (MUST match train_model.py exactly)
    features = [
        'Nitrogen', 'Phosphorus', 'Potassium', 'EC',
        'Delta_N', 'Delta_P', 'Delta_K', 'Delta_EC'
    ]
    
    X_test = df_test[features]
    y_true = df_test['Label'] # 0 = Organic, 1 = Chemical

    # 4. Run Predictions
    print("\n--- 🧠 Running AI Predictions ---")
    # Isolation Forest outputs: 1 (Normal), -1 (Anomaly)
    raw_predictions = model.predict(X_test)

    # 5. Convert Predictions to match our Labels (0/1)
    # Model says -1 (Anomaly) -> We convert to 1 (Chemical)
    # Model says  1 (Normal)  -> We convert to 0 (Organic)
    y_pred = [1 if x == -1 else 0 for x in raw_predictions]

    # 6. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
    
    print("\n--- 📝 Detailed Report ---")
    print(classification_report(y_true, y_pred, target_names=['Organic (Normal)', 'Chemical (Fraud)']))

    print("--- 🔍 Confusion Matrix ---")
    # Custom print for clarity
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"True Negatives (Organic correctly identified): {tn}")
    print(f"False Positives (Organic wrongly flagged as Fraud): {fp}")
    print(f"False Negatives (Fraud missed / undetected): {fn}")
    print(f"True Positives (Fraud correctly caught): {tp}")

    # 7. Manual "Scenario" Test (To prove the Time-Series logic works)
    print("\n--- 🧪 Scenario Test: The 'Sudden Spike' ---")
    
    # Scenario A: High Nitrogen but SLOW change (Organic Compost)
    # N=120 (High), Delta_N=2 (Slow) -> Should be SAFE (or low risk)
    organic_high_n = [[120, 50, 50, 1.0,  2.0, 0.5, 0.5, 0.02]] 
    
    # Scenario B: High Nitrogen and FAST change (Chemical Urea)
    # N=120 (High), Delta_N=40 (Fast) -> Should be ANOMALY
    chemical_spike = [[120, 50, 50, 2.5, 40.0, 0.5, 0.5, 0.8]]
    
    df_scenarios = pd.DataFrame(organic_high_n + chemical_spike, columns=features)
    scenario_preds = model.predict(df_scenarios)
    
    print(f"Scenario A (Slow Release): {'✅ Organic' if scenario_preds[0] == 1 else '❌ Flagged as Anomaly'}")
    print(f"Scenario B (Fast Spike):   {'🚨 ANOMALY Detected' if scenario_preds[1] == -1 else '❌ Missed'}")

if __name__ == "__main__":
    evaluate()