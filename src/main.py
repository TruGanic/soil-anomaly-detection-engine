from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# 1. Load the Final Model (8 Features)
try:
    model = joblib.load('models/isolation_forest_final.pkl')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found. Run train_model.py first.")

# 2. In-Memory Database to store "Previous Readings"
# Structure: { "farm_101": { "N": 40, "P": 12, "K": 20, "EC": 1.2 } }
farm_memory = {}

# 3. Define Input Data Structure (Must match Sensor Data)
class SoilData(BaseModel):
    farm_id: str
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    EC: float

@app.post("/analyze_soil")
def analyze_soil(data: SoilData):
    # --- STEP 1: CALCULATE DELTAS (Speed of Change) ---
    prev_data = farm_memory.get(data.farm_id)
    
    if prev_data:
        # Calculate change from last time
        delta_n = data.Nitrogen - prev_data['N']
        delta_p = data.Phosphorus - prev_data['P']
        delta_k = data.Potassium - prev_data['K']
        delta_ec = data.EC - prev_data['EC']
        
        is_first_reading = False
    else:
        # First time seeing this farm? Assume no change (Delta = 0)
        delta_n, delta_p, delta_k, delta_ec = 0.0, 0.0, 0.0, 0.0
        is_first_reading = True

    # Update Memory for next time
    farm_memory[data.farm_id] = {
        "N": data.Nitrogen, "P": data.Phosphorus, 
        "K": data.Potassium, "EC": data.EC
    }

    # --- STEP 2: PREPARE DATA FOR AI ---
    # The order MUST match exactly what you used in train_model.py
    features = [
        'Nitrogen', 'Phosphorus', 'Potassium', 'EC',
        'Delta_N', 'Delta_P', 'Delta_K', 'Delta_EC'
    ]
    
    input_df = pd.DataFrame([[
        data.Nitrogen, data.Phosphorus, data.Potassium, data.EC,
        delta_n, delta_p, delta_k, delta_ec
    ]], columns=features)

    # --- STEP 3: PREDICT ---
    try:
        # -1 = Anomaly, 1 = Normal
        prediction = model.predict(input_df)[0]
        confidence = model.decision_function(input_df)[0]
        
        is_anomaly = (prediction == -1)
        
        # Grading Logic (Simple Credit Score)
        # Start with 100. Subtract points for anomalies or high EC.
        score = 100
        if is_anomaly: score -= 40
        if data.EC > 2.0: score -= 20
        final_score = max(0, score)

        return {
            "farm_id": data.farm_id,
            "status": "CRITICAL ANOMALY" if is_anomaly else "COMPLIANT",
            "organic_score": final_score,
            "details": {
                "is_first_reading": is_first_reading,
                "anomalies_found": is_anomaly,
                "confidence_score": round(confidence, 4) # Lower/Negative is worse
            },
            "sensor_summary": {
                "Nitrogen": f"{data.Nitrogen} (Change: {round(delta_n, 2)})",
                "EC": f"{data.EC} (Change: {round(delta_ec, 2)})"
            }
        }

    except Exception as e:
        return {"error": str(e)}

# To Run: uvicorn src.main:app --reload