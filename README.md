# CSC-7700
Custom MLP Engine for Classification & Regression

This repository contains a custom-built Multilayer Perceptron (MLP) engine implemented from scratch in Python and applied to two tasks:

MNIST Handwritten Digit Classification

Vehicle MPG Prediction (Regression)

The project was developed to demonstrates the adaptability of neural networks by tackling both classification and regression with the same core engine.

📌 Project Overview

The MLP engine was built from the ground up using NumPy and Pandas, incorporating:

Custom activation functions (ReLU, Sigmoid, Tanh, Softmax, Softplus, Mish, Linear)

Loss functions (Cross-Entropy, Mean Squared Error)

Regularization techniques: L2 penalty and dropout

Optimization strategies: Mini-batch training, Glorot weight initialization, RMSProp, and Early Stopping

The engine is modular, making it reusable for a variety of supervised learning tasks.

📊 Datasets
1. MNIST Dataset

Source: OpenML - mnist_784

Task: Image classification (digits 0–9)

Network Architecture:

784 input neurons (28×28 pixels)

Hidden layers: 128 (ReLU), 64 (ReLU)

Output layer: 10 neurons (Softmax)

Performance: 94.79% test accuracy

2. Auto MPG Dataset

Source: UCI Machine Learning Repository

Task: Regression (predicting miles per gallon)

Network Architecture:

Input: Standardized automotive features (e.g., cylinders, displacement, horsepower, weight)

Hidden layers: 64 (ReLU), 32 (ReLU)

Output: 1 neuron (Linear activation)

Performance: Testing loss of 0.109 (MSE normalized)
