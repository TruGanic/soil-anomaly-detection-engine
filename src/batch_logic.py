from datetime import datetime, timedelta

# Mock Database of "Past Alerts" for a specific batch
# In a real app, you would fetch this from SQL/MongoDB using "SELECT * FROM readings WHERE batch_id = 'XYZ'"
batch_history = [
    {"date": "2025-09-01", "status": "COMPLIANT", "score": 100},
    {"date": "2025-09-02", "status": "COMPLIANT", "score": 100},
    {"date": "2025-09-03", "status": "COMPLIANT", "score": 100},
    {"date": "2025-09-15", "status": "CRITICAL ANOMALY", "score": 0},  # <--- Chemical Spike!
    {"date": "2025-09-16", "status": "COMPLIANT", "score": 90},       # Recovered, but trust is lost
    # ... imagine 3 months of data ...
]

def calculate_final_batch_score(readings):
    """
    Calculates the Final Organic Grade for a batch from Start to Harvest.
    """
    final_trust_score = 100
    critical_events = 0
    warning_events = 0
    
    print("\n--- BATCH VERIFICATION REPORT ---")
    
    for reading in readings:
        # Rule 1: Instant Penalties based on Daily Score
        daily_score = reading['score']
        
        if daily_score == 0: # Critical Anomaly (Chemicals)
            print(f"❌ Alert: Chemical Spike detected on {reading['date']}")
            final_trust_score -= 25 # Huge penalty
            critical_events += 1
            
        elif daily_score < 80: # Minor Warning (High EC or weird patterns)
            print(f"⚠️ Warning: Irregular patterns on {reading['date']}")
            final_trust_score -= 5
            warning_events += 1
            
    # Rule 2: The "Three Strikes" Rule (Optional)
    if critical_events >= 3:
        final_trust_score = 0
        print("⛔ AUTOMATIC FAIL: Too many chemical violations.")

    # Rule 3: Cap the score (cannot go below 0)
    final_trust_score = max(0, final_trust_score)

    # --- FINAL GRADING ---
    if final_trust_score >= 95:
        grade = "A+ (Premium Organic)"
        badge = "🟢 Gold Standard"
    elif final_trust_score >= 85:
        grade = "A (Verified Organic)"
        badge = "🟢 Certified"
    elif final_trust_score >= 70:
        grade = "B (Transitional/Risk)"
        badge = "🟡 Audit Required"
    else:
        grade = "F (Non-Compliant)"
        badge = "🔴 REJECTED"

    print(f"\n--- FINAL RESULTS ---")
    print(f"Final Trust Score: {final_trust_score}/100")
    print(f"Violations: {critical_events} Critical, {warning_events} Warnings")
    print(f"Batch Grade: {grade}")
    print(f"Certification Status: {badge}")
    
    return final_trust_score, grade

# Test Run
calculate_final_batch_score(batch_history)