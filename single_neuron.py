import pandas as pd
import numpy as np

df = pd.read_csv("Heart_disease_cleveland_new.csv")

X = df.drop("target", axis=1).values
y = df["target"].values

weights = np.random.randn(X.shape[1])
bias = 0.0

learning_rate = 0.000001
epochs = 100

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias

        prediction = sigmoid(z)

        error = prediction - y[i]

        loss = error ** 2
        total_loss += loss

        gradient = 2 * error * prediction * (1 - prediction)

        weights -= learning_rate * gradient * X[i]
        bias -= learning_rate * gradient

    print(f"Epoch {epoch + 1}, Loss = {total_loss:.4f}")

print("\nFinal Weights:")
print(weights)

print("\nFinal Bias:")
print(bias)

print("\nPredictions:")

for i in range(10):
    z = np.dot(X[i], weights) + bias

    probability = sigmoid(z)

    predicted_class = 1 if probability >= 0.5 else 0

    print(
        f"Actual={y[i]}  "
        f"Probability={probability:.4f}  "
        f"Predicted={predicted_class}"
    )