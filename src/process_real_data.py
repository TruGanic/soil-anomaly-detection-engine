import pandas as pd
import numpy as np
import os

def generate_complete_dataset():
    input_file = 'Crop_recommendation.csv'
    output_dir = 'dataset'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading real data from {input_file}...")
    try:
        df_real = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please put it in the project root.")
        return

    # --- STEP 1: PREPARE REAL DATA (ORGANIC/NORMAL) ---
    # We use the real N, P, K from the dataset to represent "Current Reading"
    organic_data = []
    
    print("Processing real data and simulating 'Time History'...")
    
    for index, row in df_real.iterrows():
        # 1. Current Values (Real from CSV)
        curr_n = row['N']
        curr_p = row['P']
        curr_k = row['K']
        
        # 2. Simulate EC (Missing in CSV)
        # Higher nutrients = Higher EC. Base 0.5 + noise.
        curr_ec = 0.5 + ((curr_n + curr_p + curr_k) / 400) + np.random.normal(0, 0.05)
        curr_ec = round(max(0.2, curr_ec), 2)

        # 3. Simulate "Previous Reading" to calculate Deltas
        # For ORGANIC soil, changes are slow. Previous value is very close to current.
        # Logic: Previous = Current +/- small random noise (0-5 ppm)
        prev_n = curr_n + np.random.normal(0, 2)
        prev_p = curr_p + np.random.normal(0, 2)
        prev_k = curr_k + np.random.normal(0, 2)
        prev_ec = curr_ec + np.random.normal(0, 0.05)

        # 4. Calculate Deltas (The "Speed" of change)
        delta_n = round(curr_n - prev_n, 2)
        delta_p = round(curr_p - prev_p, 2)
        delta_k = round(curr_k - prev_k, 2)
        delta_ec = round(curr_ec - prev_ec, 2)

        # Append [Current Values, Deltas, Label=0]
        organic_data.append([
            curr_n, curr_p, curr_k, curr_ec, 
            delta_n, delta_p, delta_k, delta_ec, 
            0 # Label 0 = Organic/Normal
        ])

    # --- STEP 2: GENERATE SYNTHETIC FRAUD DATA (CHEMICAL SPIKES) ---
    # We generate fake data where the "Previous Reading" was low, but "Current" is high.
    chemical_data = []
    print("Generating synthetic 'Chemical Spikes'...")

    for _ in range(500): # Create 500 anomaly examples
        # 1. Current Values (Spiked - e.g., Urea dumping)
        curr_n = np.random.normal(180, 40) # High N
        curr_p = np.random.normal(70, 20)
        curr_k = np.random.normal(70, 20)
        curr_ec = np.random.normal(3.5, 0.5) # High EC

        # 2. Previous Values (Normal)
        # To simulate a SUDDEN spike, we pretend the previous reading was low (normal)
        prev_n = np.random.normal(40, 10) 
        prev_p = np.random.normal(20, 10)
        prev_k = np.random.normal(20, 10)
        prev_ec = np.random.normal(1.0, 0.2)

        # 3. Calculate Deltas (These will be LARGE numbers)
        delta_n = round(curr_n - prev_n, 2)  # e.g., 180 - 40 = +140 spike
        delta_p = round(curr_p - prev_p, 2)
        delta_k = round(curr_k - prev_k, 2)
        delta_ec = round(curr_ec - prev_ec, 2)

        chemical_data.append([
            curr_n, curr_p, curr_k, curr_ec, 
            delta_n, delta_p, delta_k, delta_ec, 
            1 # Label 1 = Chemical Anomaly
        ])

    # --- STEP 3: SAVE DATASETS ---
    columns = [
        'Nitrogen', 'Phosphorus', 'Potassium', 'EC',
        'Delta_N', 'Delta_P', 'Delta_K', 'Delta_EC', 
        'Label'
    ]

    df_org = pd.DataFrame(organic_data, columns=columns)
    df_chem = pd.DataFrame(chemical_data, columns=columns)

    # Train Data: Use most of the organic data
    df_org.to_csv('dataset/train_data_final.csv', index=False)

    # Test Data: Mix of Organic and Chemical
    df_mixed = pd.concat([df_org.sample(200), df_chem]).sample(frac=1)
    df_mixed.to_csv('dataset/test_data_final.csv', index=False)

    print(f"✅ Success! Datasets created in 'dataset/'.")
    print(f"   - Features: {columns}")

if __name__ == "__main__":
    generate_complete_dataset()