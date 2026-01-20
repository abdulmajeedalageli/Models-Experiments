# PyTorch MLP & Data Engineering Experiments

This repository contains a collection of applied machine learning experiments, focusing on two key areas: building custom Neural Networks from scratch using PyTorch, and implementing robust Data Preprocessing pipelines for real-world datasets.

## 📌 Overview

The project demonstrates end-to-end ML workflows including advanced data preprocessing, feature engineering, and systematic experimentation with custom PyTorch MLPs. The notebooks explore two distinct binary classification problems: predicting wildfire risk from weather data and diagnosing diabetes from health indicators.

## 📂 Repository Contents

### 1. Data Preprocessing & Feature Engineering
**File:** `preprocessing datasets.ipynb`

This module demonstrates advanced data cleaning and analytical techniques required before model training. It explores two distinct datasets with comprehensive statistical analysis and domain-specific transformations.

#### Fire Weather Dataset
- **Source:** California Weather & Fire Dataset (2015-present)
- **Target Variable:** `FIRE_START_DAY` (Binary: Fire occurred or not)
- **Final Shape:** 3,665 rows × 15 features

**Key Preprocessing Steps:**
- **Temporal Filtering:** Retained only 2015+ data to focus on recent climate patterns
- **Log Transformations:** Applied log(1+x) to `PRECIPITATION`, `LAGGED_PRECIPITATION`, and `AVG_WIND_SPEED` to handle zero-inflated distributions (skewness > 8.0)
- **Cyclical Encoding:** Converted `DAY_OF_YEAR` into sine/cosine features so the model understands temporal continuity (Dec 31 ≈ Jan 1)
- **Multicollinearity Removal:** Dropped `MONTH` (correlation 1.0 with DAY_OF_YEAR) and `WIND_TEMP_RATIO` (correlation 0.95 with AVG_WIND_SPEED)
- **One-Hot Encoding:** Transformed `SEASON` into binary dummy variables
- **Feature Insights:** Discovered that `MIN_TEMP` (overnight warmth) is a stronger predictor than `MAX_TEMP`, and that lagged features (yesterday's weather) outperform current-day measurements

#### Diabetes Health Indicators Dataset
- **Source:** CDC BRFSS Survey (253,680 respondents)
- **Target Variable:** `Diabetes_binary` (Binary: Diabetic/Pre-diabetic vs Healthy)
- **Final Shape:** 5,000 rows × 17 features (balanced)

**Key Preprocessing Steps:**
- **Class Balancing:** Downsampled from 253,680 to 5,000 rows with perfect 50/50 class distribution to address severe 6.2:1 imbalance
- **Feature Pruning:** Removed 5 low-utility features (`CholCheck`, `AnyHealthcare`, `Stroke`, `NoDocbcCost`, `HvyAlcoholConsump`) based on Random Forest importance scores < 0.02
- **Standardization:** Applied StandardScaler to 7 continuous features (`BMI`, `Age`, `Income`, `GenHlth`, `MentHlth`, `PhysHlth`, `Education`)
- **PCA Analysis:** Tested dimensionality reduction but rejected it due to only 25.5% variance explained in first 2 components
- **Duplicate Handling:** Identified and retained 24,206 duplicates as they represent distinct individuals with identical survey responses

**Advanced Techniques:**
- Correlation heatmaps to detect multicollinearity (threshold: 0.70+)
- Skewness and kurtosis analysis for distribution assessment
- Random Forest feature importance ranking for pruning decisions
- Age × BMI interaction analysis revealing synergistic risk effects
- Probability-based binary feature analysis for risk stratification

### 2. Custom MLP Implementation
**File:** `MLP.ipynb`

A custom Multi-Layer Perceptron (MLP) built from scratch using PyTorch with 5 systematic experiments comparing different activation functions, regularization strategies, and optimization techniques.

#### Experiment 1: Baseline ReLU Model
**Architecture:**
- 3 Hidden Layers with configurable sizes (default: 64 → 32 → 16)
- Batch Normalization after each hidden layer
- ReLU activation functions
- Sigmoid output for binary classification

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross Entropy with Logits
- Early Stopping: Patience of 5 epochs with delta=0.001
- Batch Size: 64

**Evaluation Metrics:**
- ROC Curve & AUC Score
- Confusion Matrix visualization
- Classification Report (Precision, Recall, F1-Score)

#### Experiment 2: Tanh + Cross-Validation
**Key Changes:**
- **Activation:** Tanh (instead of ReLU)
- **7-Fold Cross-Validation:** Trains on all folds and selects best model by validation loss
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.1, patience=3)
- **Weight Decay:** Added L2 regularization (0.0001 for moderate model)
- **Three Model Configurations:**
  - **Basic:** h1=64, h2=32, h3=16, lr=0.001, no weight decay
  - **Moderate:** h1=128, h2=64, h3=32, lr=0.001, weight_decay=0.0001
  - **Aggressive:** h1=256, h2=128, h3=64, lr=0.01, weight_decay=0.001

#### Experiment 3: SiLU (Swish) Activation
**Key Changes:**
- **Activation:** SiLU/Swish function (smooth, non-monotonic)
- Same cross-validation and hyperparameter robustness testing as Experiment 2
- Compared performance across Basic/Moderate/Aggressive configurations

#### Experiment 4: GELU Activation
**Key Changes:**
- **Activation:** GELU (Gaussian Error Linear Unit, common in transformers)
- Same systematic testing framework as Experiments 2 & 3

#### Experiment 5: Recall-Optimized Model
**Objective:** Maximize recall for positive class (fire/diabetes detection)

**Key Changes:**
- **Class-Weighted Loss:** Automatically calculated `pos_weight` based on class imbalance in each fold
- **Lower Decision Threshold:** Changed from 0.5 to 0.3 to increase sensitivity
- **Architecture:** Used Basic configuration (64-32-16) as it showed best generalization
- **Results:** Achieved 92% recall on Fire dataset and 95% recall on Diabetes dataset

**Common Training Features Across All Experiments:**
- Custom `Dataset` class for PyTorch DataLoader integration
- Custom `EarlyStopping` class with patience and delta threshold
- Dynamic model initialization for each cross-validation fold (prevents weight carryover)
- Evaluation on held-out test set (20% split, random_state=42)

## 🛠️ Technologies Used

- **Python 3.x**
- **PyTorch** - Neural network framework
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Preprocessing, metrics, and Random Forest
- **Matplotlib & Seaborn** - Visualization
- **UCI ML Repository** - Diabetes dataset source

## 📊 Key Results Summary

### Fire Dataset Performance
- **Best Model:** Moderate configuration with Tanh activation
- **AUC Score:** ~0.75-0.77 across experiments
- **Key Finding:** Lagged precipitation is more predictive than current rainfall

### Diabetes Dataset Performance  
- **Best Model:** Recall-optimized GELU model
- **AUC Score:** ~0.73-0.75 across experiments
- **Key Finding:** BMI, Age, and GenHlth are top 3 predictive features

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn ucimlrepo

Running the Notebooks
Start with preprocessing datasets.ipynb to generate clean datasets

Run MLP.ipynb to train and evaluate models

Processed datasets are saved as:

Processed_Fire_Dataset.csv

Processed_Diabetes_dataset.csv


📝 Notes
All experiments use random_state=42 for reproducibility

The notebook includes extensive inline documentation explaining each decision

Cross-validation ensures robust model selection across data splits

Feature engineering decisions are statistically justified with correlation/importance analysis

📄 License
This project is open source and available for educational purposes.



Note: This is a learning project demonstrating ML engineering best practices including systematic experimentation, proper validation strategies, and principled feature engineering.
