PyTorch MLP & Data Engineering Experiments 
This repository contains a collection of applied machine learning experiments, focusing on two key areas: building custom Neural Networks from scratch using PyTorch, and implementing robust Data Preprocessing pipelines for real-world datasets.

Repository Contents
1. Data Preprocessing & Feature Engineering
File: preprocessing datasets.ipynb

This module demonstrates advanced data cleaning and analytical techniques required before model training. It explores two distinct datasets (Weather/Fire Data and Health Indicators).

Key Techniques:

Data Cleaning: Handling missing values, outlier detection, and duplicate removal.

Encoding: transforming categorical variables using LabelEncoder.

Feature Selection: Using a Random Forest Classifier to determine feature importance.

Visualization: Correlation matrices and heatmaps using Seaborn to understand feature relationships.

2. Custom MLP Implementation
File: MLP.ipynb

A custom Multi-Layer Perceptron (MLP) built dynamically using PyTorch. This is not a pre-built wrapper but a manual implementation of deep learning architecture.

Architecture:

Dynamic Layers: Configurable input, hidden, and output sizes.

Regularization: Implements Batch Normalization (BatchNorm1d) to stabilize learning.

Activation: ReLU for hidden layers and Sigmoid for binary classification output.

Training Loop:

Custom training loop with Adam Optimizer.

Loss calculation using Binary Cross Entropy.

Evaluation Metrics:

ROC Curve & AUC Score.

Confusion Matrix visualization.

Classification Report (Precision, Recall, F1-Score).
