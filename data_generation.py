import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
data_size = 1000

# Generate synthetic features
age = np.random.randint(18, 80, data_size)
gender = np.random.choice(['Male', 'Female'], data_size)
hemoglobin = np.random.normal(13.5, 2.0, data_size)  # g/dL
rbc_count = np.random.normal(4.7, 0.7, data_size)     # million cells/mcL
mcv = np.random.normal(90, 10, data_size)             # fL
wbc_count = np.random.normal(7.0, 2.0, data_size)     # thousand cells/mcL
platelet_count = np.random.normal(250, 50, data_size) # thousand/mcL

# Generate target: 1 = Anemia, 0 = No Anemia
# Simple rule: low hemoglobin or low RBC increases anemia risk
anemia = ((hemoglobin < 12) | (rbc_count < 4.2)).astype(int)

# Introduce some missing values
for arr in [hemoglobin, rbc_count, mcv, wbc_count, platelet_count]:
    mask = np.random.rand(data_size) < 0.05
    arr[mask] = np.nan

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Hemoglobin': hemoglobin,
    'RBC_Count': rbc_count,
    'MCV': mcv,
    'WBC_Count': wbc_count,
    'Platelet_Count': platelet_count,
    'Anemia': anemia
})

# Save to CSV
df.to_csv('synthetic_anemia_data.csv', index=False)
print('Synthetic data saved to synthetic_anemia_data.csv') 