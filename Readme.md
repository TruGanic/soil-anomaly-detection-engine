# Soil Anomaly Detection Engine

The **Soil Anomaly Detection Engine** is an AI-powered system designed to detect chemical anomalies in soil, specifically focusing on identifying "chemical spiking" (e.g., sudden dumping of urea or other fertilizers) versus natural "organic" nutrient variations.

It uses an **Isolation Forest** model to analyze not just the current nutrient levels ($N, P, K, EC$) but also the **rate of change (Deltas)** from previous readings. This allows the system to differentiate between a naturally high-nutrient soil (slow change) and an artificial spike (fast change).

## ✨ Features

-   **Real-Time Anomaly Detection**: Instantly flags readings that deviate from organic patterns.
-   **Delta Analysis**: Calculates the speed of change for Nitrogen, Phosphorus, Potassium, and Electrical Conductivity (EC).
-   **Organic Score**: Assigns a credit-score-like rating (0-100) to soil health.
-   **REST API**: Exposes endpoints for analysis via FastAPI.
-   **Synthetic Data Generation**: Simulates realistic organic and chemical fraud scenarios for training.

## 📂 Repository Structure

```text
soil-anomaly-detection-engine/
├── dataset/                # Generated CSV datasets (train/test)
├── models/                 # Saved machine learning models (.pkl)
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── main.py             # FastAPI application entry point
│   ├── process_real_data.py# Data generation script (Organic vs Chemical)
│   ├── train_model.py      # Model training script
│   └── evaluate_model.py   # Model evaluation and testing script
├── requirements.txt        # Python dependencies
└── Readme.md               # Project documentation
```

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd soil-anomaly-detection-engine
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Usage Workflow

Follow these steps to generate data, train the model, and run the system.

### 1. Data Generation
Generate synthetic datasets (Training & Testing) that simulate both organic soil patterns and chemical spikes. This script reads `Crop_recommendation.csv` (if available) or generates synthetic baselines.

```bash
python src/process_real_data.py
```
*Output: Creates `dataset/train_data_final.csv` and `dataset/test_data_final.csv`.*

### 2. Train Model
Train the Isolation Forest model on the generated dataset.

```bash
python src/train_model.py
```
*Output: Saves the trained model to `models/isolation_forest_final.pkl`.*

### 3. Evaluate Model
Run the evaluation script to test accuracy, view the confusion matrix, and simulate specific fraud scenarios (e.g., "Sudden Urea Spike").

```bash
python src/evaluate_model.py
```
*Output: Displays accuracy score, classification report, and scenario test results.*

### 4. Run API Server
Start the FastAPI server to accept real-time soil data requests.

```bash
uvicorn src.main:app --reload
```
*The API will be available at `http://127.0.0.1:8000`.*

## 🔌 API Documentation

### Analyze Soil
**POST** `/analyze_soil`

Analyzes a soil reading and determines if it is organic or an anomaly based on previous history (in-memory).

**Request Body:**
```json
{
  "farm_id": "farm_101",
  "Nitrogen": 120.0,
  "Phosphorus": 45.0,
  "Potassium": 50.0,
  "EC": 1.2
}
```

**Response:**
```json
{
  "farm_id": "farm_101",
  "status": "COMPLIANT",
  "organic_score": 90,
  "details": {
    "is_first_reading": false,
    "anomalies_found": false,
    "confidence_score": 0.15
  },
  "sensor_summary": {
    "Nitrogen": "120.0 (Change: 2.0)",
    "EC": "1.2 (Change: 0.05)"
  }
}
```

## 📜 License
[MIT License](LICENSE) (or specify your license here)
