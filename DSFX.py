import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.feature_selection import f_classif, RFE
from sklearn.linear_model import LogisticRegression

def extract_features(data_path):
    # Read the CSV file
    data = pd.read_csv(data_path)

    # Separate the labels from the features
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    # Calculate statistical features
    min_val = features.min(axis=1)
    max_val = features.max(axis=1)
    mean_val = features.mean(axis=1)
    std_val = features.std(axis=1)
    
    # Calculate quartiles (q1 to q4)
    q1 = features.quantile(0.25, axis=1)
    q2 = features.quantile(0.50, axis=1)
    q3 = features.quantile(0.75, axis=1)
    q4 = features.quantile(1.0, axis=1)
    
    # Calculate skewness and kurtosis
    skewness = features.apply(skew, axis=1)
    kurt = features.apply(kurtosis, axis=1)
    
    # Calculate entropy
    entropy = -np.nansum(features * np.log(features), axis=1)
    
    # Calculate FFT domain features (min, max, mean, std)
    fft_features = np.abs(fft(features, axis=1))
    fft_min = fft_features.min(axis=1)
    fft_max = fft_features.max(axis=1)
    fft_mean = fft_features.mean(axis=1)
    fft_std = fft_features.std(axis=1)

    # Create a DataFrame for the extracted features
    extracted_features = pd.DataFrame({
        'Label': labels,
        'Min': min_val,
        'Max': max_val,
        'Mean': mean_val,
        'Std': std_val,
        'Q1': q1,
        'Q2': q2,
        'Q3': q3,
        'Q4': q4,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Entropy': entropy,
        'FFT_Min': fft_min,
        'FFT_Max': fft_max,
        'FFT_Mean': fft_mean,
        'FFT_Std': fft_std
    })

    return extracted_features

# Example usage:
data_path = '/content/drive/MyDrive/WPI/gprMax_simulations/10000_simulation_results/training_dataset_10000_1l.csv'
extracted_features = extract_features(data_path)
