"""
Bacterial Growth Rate Analysis
lin.su@qmul.ac.uk, 2025/02/21

This script is designed to analyze bacterial growth from optical density (OD) measurements over time. 
It automatically detects the exponential growth phase within a given time range, calculates the growth rate 
and doubling time, and plots the growth curve with the exponential fit. 
An annotation of the growth rate is added to each figure.

The input CSV file should have the following format:
- The first column should contain time points.
- Subsequent columns should contain OD measurements for each replicate.
- Column headers should be included at the top of the CSV file, with the first column header typically being 'Time' or similar.

Example CSV file format:
Time,Replicate1,Replicate2,Replicate3
0,0.1,0.11,0.09
1,0.2,0.21,0.19
2,0.4,0.42,0.39
...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def detect_exponential_phase(time, od_values, min_window_size=3, max_window_size=7, r_squared_threshold=0.8):
    best_r_squared = 0
    best_slope = 0
    best_start, best_end, best_window_size = 0, min_window_size, min_window_size

    for window_size in range(min_window_size, max_window_size + 1):
        for start in range(len(time) - window_size + 1):
            end = start + window_size
            slope, intercept, r_value, _, _ = linregress(time[start:end], np.log(od_values[start:end]))
            current_r_squared = r_value**2
            # Prioritize higher slope, ensuring R² is above the threshold
            if current_r_squared > r_squared_threshold and slope > best_slope:
                best_r_squared = current_r_squared
                best_slope = slope
                best_start, best_end, best_window_size = start, end, window_size

    return best_start, best_end, best_window_size, best_slope, best_r_squared

data = pd.read_csv('datafile.csv')  # Ensure the data file is in the correct path/format

# Plot the original ln(OD) data for all replicates
plt.figure(figsize=(10, 6))
for col in data.columns[1:]:
    plt.plot(data['time'], np.log(data[col]), label=col)  # Plotting ln(OD)
plt.xlabel('Time (days)')
plt.ylabel('ln(OD)')
plt.title('Original Bacterial Growth Data (Logarithmic Scale)')
plt.legend()
plt.show()

growth_rates = []
doubling_times = []
r_squared_values = []

for col in data.columns[1:]:
    start_index, end_index, window_size, slope, r_squared = detect_exponential_phase(data['time'], data[col])
    
    # Extract the time and OD values for the exponential phase
    time_exp = data['time'][start_index:end_index]
    od_exp = data[col][start_index:end_index]

    # Perform the linear regression on the exponential phase
    slope, intercept, r_value, _, _ = linregress(time_exp, np.log(od_exp))
    r_squared = r_value**2

    # Store the growth rate and doubling time
    growth_rate = slope
    doubling_time = np.log(2) / growth_rate
    growth_rates.append(growth_rate)
    doubling_times.append(doubling_time)
    r_squared_values.append(r_squared)

    # Plot the original ln(OD) values and the exponential fit
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], np.log(data[col]), 'o', label=f'Original ln(OD) - {col}')
    fit_line = slope * time_exp + intercept
    plt.plot(time_exp, fit_line, 'r--', label=f'Exponential Fit (R² = {r_squared:.2f})')
    
    plt.xlabel('Time (days)')
    plt.ylabel('ln(OD)')
    plt.title(f'Exponential Growth Fit for {col} (Logarithmic Scale)')
    
    # Annotate the figure with the growth rate
    plt.text(0.05, 0.95, f'Growth rate: {growth_rate:.4f} per day', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.legend()
    plt.show()

    print(f"{col} - Growth rate: {growth_rate:.4f} per day, Doubling time: {doubling_time:.2f} days, R²: {r_squared:.2f}")

mean_growth_rate = np.mean(growth_rates)
sd_growth_rate = np.std(growth_rates, ddof=1)
mean_doubling_time = np.mean(doubling_times)
sd_doubling_time = np.std(doubling_times, ddof=1)
mean_r_squared = np.mean(r_squared_values)
sd_r_squared = np.std(r_squared_values, ddof=1)

print("\nOverall Results:")
print(f"Mean Growth Rate: {mean_growth_rate:.4f} per day, SD: {sd_growth_rate:.4f}")
print(f"Mean Doubling Time: {mean_doubling_time:.2f} days, SD: {sd_doubling_time:.2f}")
print(f"Mean R²: {mean_r_squared:.2f}, SD: {sd_r_squared:.2f}")
