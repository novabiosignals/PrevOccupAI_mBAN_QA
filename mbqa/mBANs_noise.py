# main libraries
import numpy as np
import pandas as pd

# local imports
from .adc_a import adc_to_acceleration

# Load the two matrices (each shape: n_samples x 3) using pandas.read_csv
data1_df = pd.read_csv(
    "/Users/goncalobarros/Documents/projects/New_MB_QA/2025-07-11_10-59-53/opensignals_588E81A249C7_2025-07-11_10-59-55.txt",
    comment='#',
    delimiter='\t',
    header=None,
    usecols=[3, 4, 5],
    names=['data1_x', 'data1_y', 'data1_z']
)

data2_df = pd.read_csv("/Users/goncalobarros/Documents/projects/New_MB_QA/2025-07-11_10-59-53/opensignals_886B0FABF703_2025-07-11_10-59-53.txt", 
    comment='#',
    delimiter='\t',
    header=None,
    usecols=[3, 4, 5],
    names=['data2_x', 'data2_y', 'data2_z']
)

# Convert to numpy arrays for further processing
data1_adc = data1_df.values  # shape: (n_samples, 3)
print("Data1 ADC shape:", data1_adc.shape)

data2_adc = data2_df.values  # shape: (n_samples, 3)
print("Data2 ADC shape:", data2_adc.shape)

# Ensure both matrices have the same number of samples
if data1_adc.shape[0] != data2_adc.shape[0]:
    print("Warning: The two datasets have different number of samples.")
    # Trim the longer dataset to match the shorter one
    min_samples = min(data1_adc.shape[0], data2_adc.shape[0])
    data1_adc_trimmed = data1_adc[:min_samples, :]
    data2_adc_trimmed = data2_adc[:min_samples, :]

# Concatenate the two matrices along the second axis (axis=1)
# This will result in a shape of (n_samples, 6) for the concatenated
data_concat = adc_to_acceleration(np.concatenate((data1_adc_trimmed, data2_adc_trimmed), axis=1))

# Calculate noise variance for each of '6' axis
print("Concatenated data shape:", data_concat.shape)  # Should be (n_samples, 6)

var = np.var(data_concat, axis=0, ddof=0)  # Variance for each axis
print("Variance per axis:", var)

# Define a variance limit based on the noise variance
# This is a threshold to identify significant deviations from noise
k = 8          # between 5 and 10, depending on how conservative you want to be

# This factor is used to set a limit for detecting significant deviations from noise
var_limit = k * var # [σx², σy², σz²] * k

# Find the minimum value between the x-axis of the first and second datasets (columns 0 and 3 after concatenation)
min_x = min(var_limit[0], var_limit[3])  # Minimum value of the x-axis variance limit
min_y = min(var_limit[1], var_limit[4])  # Minimum value of the y-axis variance limit
min_z = min(var_limit[2], var_limit[5])  # Minimum value of the z-axis variance limit

# Create a final variance limit array for each axis
min_var_limit = np.min([min_x, min_y, min_z])  # Final variance limit for each axis

print("Mininum variance limit per axis:", min_var_limit)



