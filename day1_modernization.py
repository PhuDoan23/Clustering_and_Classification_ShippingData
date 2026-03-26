import pandas as pd
import numpy as np

def modernize_dataset(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    
    np.random.seed(42) # For reproducibility
    
    # 1. Add Fuel_Type
    # 2026 maritime transition assumed distribution
    fuel_types = ['VLSFO', 'HFO', 'MGO', 'LNG', 'Methanol', 'Ammonia']
    fuel_probs = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
    df['Fuel_Type'] = np.random.choice(fuel_types, size=len(df), p=fuel_probs)
    
    # Emission factors (tons CO2 per ton of fuel)
    emission_factors = {
        'VLSFO': 3.151,
        'HFO': 3.114,
        'MGO': 3.206,
        'LNG': 2.75,
        'Methanol': 1.375,
        'Ammonia': 0.0
    }
    
    # Map emission factors
    df['Emission_Factor'] = df['Fuel_Type'].map(emission_factors)
    
    # 2. Add Carbon_Intensity_Indicator (CII)
    # Estimate specific fuel consumption (SFC) in g/kWh. Varies slightly by fuel.
    # We will use a baseline of 180 g/kWh.
    baseline_sfc_g_kwh = 180
    
    # Calculate voyage time (hours)
    voyage_time_hours = df['Distance_Traveled_nm'] / df['Speed_Over_Ground_knots']
    
    # Handle possible division by zero or infinites if speed is 0
    voyage_time_hours = voyage_time_hours.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Total energy consumed (kWh) = Power * Time
    energy_kwh = df['Engine_Power_kW'] * voyage_time_hours
    
    # Total fuel consumed (tons) = Energy * SFC / 1,000,000
    fuel_consumed_tons = energy_kwh * baseline_sfc_g_kwh / 1e6
    
    # Total CO2 emissions (tons)
    df['CO2_Emissions_tons'] = fuel_consumed_tons * df['Emission_Factor']
    
    # Estimate Ship Capacity (DWT) from Cargo Weight and Average Load Percentage
    # Capacity = Cargo_Weight / (Load Percentage / 100)
    df['Estimated_Capacity_DWT'] = np.where(df['Average_Load_Percentage'] > 0,
                                            df['Cargo_Weight_tons'] / (df['Average_Load_Percentage'] / 100),
                                            df['Cargo_Weight_tons'])
    # Avoid zero capacity
    df['Estimated_Capacity_DWT'] = df['Estimated_Capacity_DWT'].replace(0, 50000) # fallback
                                            
    # CII = CO2 Emissions (grams) / (Capacity * Distance)
    co2_grams = df['CO2_Emissions_tons'] * 1e6
    df['CII_Value'] = co2_grams / (df['Estimated_Capacity_DWT'] * df['Distance_Traveled_nm'])
    
    # Fill any NaNs with median
    df['CII_Value'] = df['CII_Value'].fillna(df['CII_Value'].median())
    
    # Assign CII Ratings (A to E) based on quantiles (A is best, E is worst)
    quantiles = df['CII_Value'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    def get_cii_rating(val):
        if val <= quantiles[0.2]: return 'A'
        elif val <= quantiles[0.4]: return 'B'
        elif val <= quantiles[0.6]: return 'C'
        elif val <= quantiles[0.8]: return 'D'
        else: return 'E'
        
    df['CII_Rating'] = df['CII_Value'].apply(get_cii_rating)
    
    # 3. Inject Sensor Noise (Gaussian)
    # Give some realistic sensor errors
    # Speed GPS noise: mean 0, std 0.1 knots
    speed_noise = np.random.normal(0, 0.1, size=len(df))
    df['Speed_Over_Ground_knots'] = np.clip(df['Speed_Over_Ground_knots'] + speed_noise, 0.1, None)
    
    # Engine Power telemetry noise: mean 0, std 50 kW
    power_noise = np.random.normal(0, 50, size=len(df))
    df['Engine_Power_kW'] = np.clip(df['Engine_Power_kW'] + power_noise, 100, None)
    
    # Draft sensor noise: mean 0, std 0.05 meters
    draft_noise = np.random.normal(0, 0.05, size=len(df))
    df['Draft_meters'] = np.clip(df['Draft_meters'] + draft_noise, 0.5, None)
    
    # Save the modernized dataset
    df.to_csv(output_path, index=False)
    print(f"Modernized dataset saved to {output_path}")
    print("\nSample of new columns:")
    print(df[['Fuel_Type', 'CII_Value', 'CII_Rating', 'Speed_Over_Ground_knots', 'Engine_Power_kW']].head())

if __name__ == "__main__":
    modernize_dataset("Ship_Performance_Dataset.csv", "Ship_Performance_Dataset_Modernized.csv")
