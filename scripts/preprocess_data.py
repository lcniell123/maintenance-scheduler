import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load raw data
print("Loading data...")
data = pd.read_excel('data/test.xlsx', engine='openpyxl')  # Adjust the path as needed
print(f"Data loaded successfully. Shape: {data.shape}")
print(f"Available columns: {data.columns}")

# Define the list of columns you want to use
# Define the list of columns you want to use
desired_columns = ['Vibration Frequency',
       'Vibration Amplitude', 'Bearing Temperature', 'Motor Temperature',
       'Belt Load', 'Torque', 'Noise Levels', 'Current and Voltage',
       'Hydraulic Pressure', 'Belt Thickness', 'Roller Condition'] # Replace with the actual column names
# Convert the desired columns to numeric, coerce errors to NaN
for column in desired_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Fill NaN values with 0
data[desired_columns] = data[desired_columns].fillna(0)

# Preprocessing: Scale the data using MinMaxScaler
print("Scaling data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[desired_columns])

# Create sequences (for time-series data)
sequence_length = 60
X, y = [], []

# Generate sequences
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  # Adjust based on the prediction target

X, y = np.array(X), np.array(y)
print(f"Generated sequences. X shape: {X.shape}, y shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the training and test sets to have 11 features
X_train = X_train.reshape((X_train.shape[0], 60, 11))
X_test = X_test.reshape((X_test.shape[0], 60, 11))

# Save the processed data
np.savetxt('data/X_train.csv', X_train.reshape(X_train.shape[0], -1), delimiter=",")  # Flatten X_train to 2D
np.savetxt('data/y_train.csv', y_train, delimiter=",")
np.savetxt('data/X_test.csv', X_test.reshape(X_test.shape[0], -1), delimiter=",")  # Flatten X_test to 2D
np.savetxt('data/y_test.csv', y_test, delimiter=",")

print("Train and test data created and saved successfully.")

