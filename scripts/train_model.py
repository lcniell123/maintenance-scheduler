import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Load your preprocessed training data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",")

# Check the shape of y_train
print(f"Original y_train shape: {y_train.shape}")

# If y_train is 1-dimensional, ensure it's a column vector
if y_train.ndim == 1:
    y_train = y_train.reshape(-1, 1)  # Reshape to make it 2D

# Reshape X_train to 3D (samples, timesteps, features)
samples = X_train.shape[0]
timesteps = 60
features = 11  # Based on the model setup
X_train = X_train.reshape((samples, timesteps, features))

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),  # Unique classes in the training set
    y=y_train.ravel()  # Use ravel to flatten for class weight computation
)

# Convert to dictionary format
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Define the model architecture
model = Sequential()
model.add(Input(shape=(timesteps, features)))  # Use Input layer
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100))  # Last LSTM layer without return_sequences
model.add(Dropout(0.3))
model.add(Dense(1))  # Output layer for regression

# Compile the model
optimizer = Adam(learning_rate=0.00001)  # Adjust the learning rate as needed
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model using class weights
history = model.fit(X_train, y_train, epochs=300, batch_size=32,
                    class_weight=class_weights, validation_split=0.2,
                    callbacks=[early_stopping])

# Plot training & validation loss values
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)  # Optional
plt.show()

# Save the model
model.save('models/saved_model.keras')
print("Model trained and saved successfully.")
