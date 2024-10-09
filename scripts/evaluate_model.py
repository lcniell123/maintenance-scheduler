import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

# Load preprocessed test data
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",")

# Check for class imbalance
y_test_distribution = pd.Series(y_test).value_counts()
print("Class distribution in y_test:")
print(y_test_distribution)

# Reshape X_test to 3D (samples, timesteps, features)
samples = X_test.shape[0]
timesteps = 60
features = 11  # Based on the model setup
X_test = X_test.reshape((samples, timesteps, features))

# Print shapes for debugging
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Load the saved model
model = load_model('models/saved_model.keras')
model.summary()  # Check the model structure

# Evaluate the model
try:
    batch_size = 1  # Set to 1 for evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
except Exception as e:
    print("Error during evaluation:", str(e))

# Make predictions
y_pred = model.predict(X_test)

# Visualize the predictions vs actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='orange')
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.show()
