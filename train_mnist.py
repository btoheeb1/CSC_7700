import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

# ---------------- Batch Generator ---------------- #
def batch_generator(train_x, train_y, batch_size):
    for i in range(0, len(train_x), batch_size):
        yield train_x[i:i+batch_size], train_y[i:i+batch_size]


# ---------------- Activation Functions ---------------- #
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.forward(x) * (1 - self.forward(x))

class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class Softmax(ActivationFunction):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x):
        return 1

class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x):
        omega = 4 * (x + 1) + 4 * np.exp(2*x) + np.exp(3*x) + np.exp(x) * (4*x + 6)
        delta = (2 * np.exp(x) + np.exp(2*x) + 2)**2
        return np.exp(x) * omega / delta


# ---------------- Loss Functions ---------------- #
class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def derivative(self, y_true, y_pred):
        return - (y_true - y_pred)

class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def derivative(self, y_true, y_pred):
        return (y_pred - y_true)

# ---------------- Layer Class with L2 Regularization ---------------- #
class Layer:
    def __init__(self, fan_in, fan_out, activation_function, dropout_rate=0.0, l2_lambda=0.001):
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.biases = np.zeros((1, fan_out))
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.dropout_mask = None

    def forward(self, h, training=True):
        self.input = h
        self.z = np.dot(h, self.weights) + self.biases
        self.output = self.activation_function.forward(self.z)

        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            self.output *= self.dropout_mask

        return self.output
    def backward(self, grad_output, learning_rate):
        if self.dropout_rate > 0:
            grad_output *= self.dropout_mask

        activation_grad = grad_output * self.activation_function.derivative(self.z)
        grad_weights = np.dot(self.input.T, activation_grad) + self.l2_lambda * self.weights
        grad_biases = np.sum(activation_grad, axis=0, keepdims=True)
        grad_input = np.dot(activation_grad, self.weights.T)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

# ---------------- Early Stopping and Training Enhancements ---------------- #
def early_stopping(val_losses, patience=5):
    if len(val_losses) > patience and val_losses[-1] > min(val_losses[-patience:]):
        return True
    return False

# ---------------- Multilayer Perceptron with RMSProp ---------------- #
class MultilayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
        self.cache = [np.zeros_like(layer.weights) for layer in self.layers]

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, loss_grad, learning_rate, rmsprop=True, beta=0.9, epsilon=1e-8):
        grad = loss_grad
        for i, layer in enumerate(reversed(self.layers)):
            grad = layer.backward(grad, learning_rate)
            if rmsprop:
                self.cache[i] = beta * self.cache[i] + (1 - beta) * (layer.weights ** 2)
                layer.weights -= (learning_rate / (np.sqrt(self.cache[i]) + epsilon)) * layer.weights

    def train(self, train_x, train_y, val_x, val_y, loss_func, learning_rate, batch_size, epochs, rmsprop=False, beta=0.9, epsilon=1e-8):
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            batch_losses = []
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                output = self.forward(batch_x, training=True)
                loss = loss_func.loss(batch_y, output)
                batch_losses.append(loss)
                loss_grad = loss_func.derivative(batch_y, output)
                self.backward(loss_grad, learning_rate, rmsprop, beta, epsilon)

            train_losses.append(np.mean(batch_losses))
            val_output = self.forward(val_x, training=False)
            val_loss = loss_func.loss(val_y, val_output)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")
        return train_losses, val_losses

# ---------------- MNIST Data Loading ---------------- #
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(np.array(y).reshape(-1, 1))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

# ---------------- MNIST Training Script ---------------- #
if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

    layers = [
        Layer(784, 128, Relu()),
        Layer(128, 64, Relu()),
        Layer(64, 10, Softmax())
    ]

    mlp = MultilayerPerceptron(layers)
    loss_function = CrossEntropy()

    train_losses, val_losses = mlp.train(
        X_train, y_train, X_val, y_val,
        loss_function, learning_rate=0.00001, batch_size=64, epochs=200
    )

    # Evaluate on the test set
    test_output = mlp.forward(X_test)
    test_accuracy = np.mean(np.argmax(test_output, axis=1) == np.argmax(y_test, axis=1))
    test_loss = loss_function.loss(y_test, test_output)
    print(f'Total Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
   
    # Plot training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Select one sample per class (0-9) and display images with predictions
    selected_indices = []
    predicted_labels = np.argmax(test_output, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    X_test_array = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    for i in range(10):
        indices = np.where(true_labels == i)[0]
        sample_index = indices[0] if len(indices) > 0 else None
        if sample_index is not None:
            selected_indices.append(sample_index)
    
    plt.figure(figsize=(10, 4))
    for idx, sample_index in enumerate(selected_indices):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(X_test_array[sample_index].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_labels[sample_index]}\nPred: {predicted_labels[sample_index]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
