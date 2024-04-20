import numpy as np
import pandas as pd
from model import gradient_descent, make_predictions, get_accuracy

# Load data
data = pd.read_csv('digit-recognizer/train.csv').to_numpy()
np.random.shuffle(data)

# Split data
X_train = data[1000:, 1:].T / 255.0
Y_train = data[1000:, 0]
X_test = data[:1000, 1:].T / 255.0
Y_test = data[:1000, 0]

# Train model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 100)

# Evaluate model
Test_predictions = make_predictions(X_test, W1, b1, W2, b2)
accuracy = get_accuracy(Test_predictions, Y_test)
print(f"\nAccuracy on test dataset: {accuracy}")


