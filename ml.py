import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# Function to extract features (same as before)
def extract_features(data_path):
    # ... (previous code) ...

    return extracted_features

# Function to train and evaluate a neural network
def train_and_evaluate_nn(data_path, top_features):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Separate the labels from the features
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    # Select the top features based on importance
    selected_features = features[top_features['Feature']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train a neural network using raw data
    raw_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    start_time = time.time()
    raw_nn.fit(X_train_scaled, y_train)
    raw_training_time = time.time() - start_time

    # Initialize and train a neural network using extracted features
    ext_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    start_time = time.time()
    ext_nn.fit(X_train_scaled, y_train)
    ext_training_time = time.time() - start_time

    # Apply PCA to the data with 2 components
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_train_scaled)

    # Initialize and train a neural network using PCA data
    pca_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    start_time = time.time()
    pca_nn.fit(pca_data, y_train)
    pca_training_time = time.time() - start_time

    # Predictions
    raw_train_preds = raw_nn.predict(X_train_scaled)
    raw_test_preds = raw_nn.predict(X_test_scaled)

    ext_train_preds = ext_nn.predict(X_train_scaled)
    ext_test_preds = ext_nn.predict(X_test_scaled)

    pca_test_data = pca.transform(X_test_scaled)
    pca_test_preds = pca_nn.predict(pca_test_data)

    # Calculate RMSE
    raw_train_rmse = np.sqrt(mean_squared_error(y_train, raw_train_preds))
    raw_test_rmse = np.sqrt(mean_squared_error(y_test, raw_test_preds))

    ext_train_rmse = np.sqrt(mean_squared_error(y_train, ext_train_preds))
    ext_test_rmse = np.sqrt(mean_squared_error(y_test, ext_test_preds))

    pca_test_rmse = np.sqrt(mean_squared_error(y_test, pca_test_preds))

    return {
        'raw_train_rmse': raw_train_rmse,
        'raw_test_rmse': raw_test_rmse,
        'ext_train_rmse': ext_train_rmse,
        'ext_test_rmse': ext_test_rmse,
        'pca_test_rmse': pca_test_rmse,
        'raw_training_time': raw_training_time,
        'ext_training_time': ext_training_time,
        'pca_training_time': pca_training_time
    }

# Example usage:
data_path = '/content/drive/MyDrive/WPI/gprMax_simulations/10000_simulation_results/training_dataset_10000_1l.csv'
extracted_features = extract_features(data_path)
top_features = feature_importance_df.head(5)

results = train_and_evaluate_nn(data_path, top_features)

# Plot RMSE and Training Time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

labels = ['Raw Train', 'Raw Test', 'Extracted Train', 'Extracted Test', 'PCA Test']
rmse_values = [results['raw_train_rmse'], results['raw_test_rmse'], results['ext_train_rmse'],
               results['ext_test_rmse'], results['pca_test_rmse']]
training_times = [results['raw_training_time'], results['ext_training_time'], results['pca_training_time']]

ax1.bar(labels, rmse_values, color=['b', 'g', 'b', 'g', 'r'])
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE Comparison')

ax2.bar(['Raw', 'Extracted', 'PCA'], training_times, color=['b', 'g', 'r'])
ax2.set_ylabel('Training Time (s)')
ax2.set_title('Training Time Comparison')

plt.tight_layout()
plt.show()
