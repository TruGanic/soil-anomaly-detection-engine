from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timedelta
import joblib
import pandas as pd
import os

app = FastAPI()

# --- 1. LOAD AI MODEL ---
try:
    model = joblib.load('models/isolation_forest_final.pkl')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found. Run train_model.py first.")

# --- 2. MONGODB CONNECTION ---
from dotenv import load_dotenv
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['test'] # CHANGE THIS to your actual database name

crop_batches_col = db['cropbatches']
input_logs_col = db['inputlogs']
soil_readings_col = db['soilreadings'] 

# --- 3. INPUT SCHEMA ---
class SoilData(BaseModel):
    sensor_id: str  # Matches the simulator payload exactly
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    EC: float

@app.post("/analyze_soil")
def analyze_soil(data: SoilData):
    # --- STEP 1: VERIFY HARDWARE & GET BATCH ---
    active_batch = crop_batches_col.find_one({
        "sensorId": data.sensor_id, 
        "status": "Active"
    })
    
    if not active_batch:
        return {
            "status": "IDLE", 
            "organic_score": "N/A", 
            "reason": "No active crop planted for this sensor."
        }
        
    current_batch_id = active_batch["batchId"]
    current_percentage = active_batch.get("currentOrganicLevel", 100)

    # --- STEP 2: CALCULATE TIME-SERIES DELTAS ---
    prev_reading = soil_readings_col.find_one(
        {"batchId": current_batch_id},
        sort=[("createdAt", -1)] 
    )
    
    if prev_reading:
        delta_n = round(data.Nitrogen - prev_reading['Nitrogen'], 2)
        delta_p = round(data.Phosphorus - prev_reading['Phosphorus'], 2)
        delta_k = round(data.Potassium - prev_reading['Potassium'], 2)
        delta_ec = round(data.EC - prev_reading['EC'], 2)
    else:
        delta_n, delta_p, delta_k, delta_ec = 0.0, 0.0, 0.0, 0.0

    # --- STEP 3: RUN AI ---
    features = [
        'Nitrogen', 'Phosphorus', 'Potassium', 'EC',
        'Delta_N', 'Delta_P', 'Delta_K', 'Delta_EC'
    ]
    
    input_df = pd.DataFrame([[
        data.Nitrogen, data.Phosphorus, data.Potassium, data.EC,
        delta_n, delta_p, delta_k, delta_ec
    ]], columns=features)

    try:
        prediction = model.predict(input_df)[0]
        confidence_score = model.decision_function(input_df)[0]
        is_anomaly = bool(prediction == -1)

        # Save CURRENT reading for the next cycle
        soil_readings_col.insert_one({
            "batchId": current_batch_id,
            "sensorId": data.sensor_id,
            "Nitrogen": data.Nitrogen,
            "Phosphorus": data.Phosphorus,
            "Potassium": data.Potassium,
            "EC": data.EC,
            "isAnomaly": is_anomaly, # Save this to easily count violations later
            "createdAt": datetime.utcnow()
        })

        # --- STEP 4: ZERO-TRUST CROSS-CHECKING ---
        if is_anomaly:
            forty_eight_hours_ago = datetime.utcnow() - timedelta(hours=48)
            recent_logs = list(input_logs_col.find({
                "batchId": current_batch_id,
                "date": {"$gte": forty_eight_hours_ago}
            }))

            if confidence_score < -0.15:
                penalty = 20
                reason = "Severe unexplained chemical spike detected."
            elif confidence_score < -0.05:
                penalty = 10
                reason = "Moderate unexplained nutrient spike detected."
            else:
                penalty = 5
                reason = "Minor irregular nutrient pattern detected."

            if any(log.get('inputCategory') == 'Organic Fertilizer' for log in recent_logs):
                reason += " (Zero-Trust Alert: Spike contradicts recent 'Organic Fertilizer' log)"

            new_percentage = max(0, current_percentage - penalty)
            crop_batches_col.update_one(
                {"batchId": current_batch_id},
                {"$set": {
                    "currentOrganicLevel": new_percentage,
                    "updatedAt": datetime.utcnow()
                }}
            )

            return {
                "status": "CRITICAL ANOMALY",
                "organic_score": new_percentage,
                "reason": reason
            }

        return {
            "status": "COMPLIANT",
            "organic_score": current_percentage,
            "reason": "Normal expected soil behavior."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))