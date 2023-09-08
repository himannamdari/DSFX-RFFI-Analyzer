import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Function to extract features (same as before)
def extract_features(data_path):
    # ... (previous code) ...

    return extracted_features

# Function to train a Random Forest model and get feature importances
def train_random_forest_and_get_feature_importances(extracted_features):
    # Separate features and labels
    X = extracted_features.drop('Label', axis=1)
    y = extracted_features['Label']

    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X, y)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Create a DataFrame to associate feature names with importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

# Example usage:
data_path = '/content/drive/MyDrive/WPI/gprMax_simulations/10000_simulation_results/training_dataset_10000_1l.csv'
extracted_features = extract_features(data_path)

# Train Random Forest model and get feature importances
feature_importance_df = train_random_forest_and_get_feature_importances(extracted_features)

# Plot the top five important features
top_features = feature_importance_df.head(5)
plt.figure(figsize=(10, 6))
plt.bar(top_features['Feature'], top_features['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 5 Important Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
