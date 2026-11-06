# Swarm Behaviour Classification

This repository contains an end-to-end Machine Learning and Deep Learning pipeline to classify swarm behaviors using the **Swarm Behaviour Classification** dataset from Kaggle. The study compares classical classifiers and sequential models under two distinct feature engineering setups: **with** and **without** a time-series sliding window approach.

## About the Dataset
- **Source**: Kaggle (`deepcontractor/swarm-behaviour-classification`)
- **Size**: 23,309 samples, 2,401 columns (2400 features describing coordinates, velocity, acceleration, alignment, and separation of 200 distinct agents, plus the binary target label `Swarm_Behaviour`).
- **Target Class Balance**:
  - Class `0` (Non-Flocking): 15,355 samples (~66%)
  - Class `1` (Flocking): 7,954 samples (~34%)

## Project Workflow
1. **Exploratory Data Analysis (EDA)**: Analyzed agent patterns, distributions, and spatial metrics over time.
2. **Feature Engineering**:
   - **Standard Pipeline**: Scaling followed by Dimensionality Reduction via Principal Component Analysis (PCA).
   - **Sequential/Temporal Pipeline**: Reframing the binary target status as a supervised time-series tracking task using a 7-step lagging Sliding Window.
3. **Model Development**: Evaluated Naive Bayes, Hidden Markov Models (HMM), Support Vector Machines (SVM), and Long Short-Term Memory (LSTM) Networks.

## Feature Engineering & Preprocessing

### 1. Standard PCA (No Sliding Window)
- **Normalization**: Features are standardized using `StandardScaler`.
- **Dimensionality Reduction**: PCA is applied to compress 2400 dimensions. Selecting $k=400$ principal components preserves **~88.5%** of the dataset's cumulative variance.

### 2. Sliding Window Method
- Reframes the classification as a temporal context task:
  - Input: $X = [y(t-7), y(t-6), ..., y(t-1)]$
  - Output: $y = y(t)$

## Models Evaluated & Performance

| Model | Strategy | Train Accuracy | Test Accuracy | Test MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Bernoulli Naive Bayes** | Without Sliding Window | 87.07% | 87.41% | 0.1259 |
| **Bernoulli Naive Bayes** | With Sliding Window | 65.74% | 66.25% | 0.3375 |
| **Gaussian HMM** | Without Sliding Window | 41.14% | 41.92% | 0.5808 |
| **Gaussian HMM** | With Sliding Window | 54.80% | 55.56% | 0.4444 |
| **Support Vector Machine (SVM)** | Without Sliding Window (Linear) | **92.36%** | **90.13%** | **0.0987** |
| **Support Vector Machine (SVM)** | With Sliding Window (RBF) | 65.74% | 66.25% | 0.3375 |
| **LSTM Neural Network** | Without Sliding Window | 65.88% | 65.87% | 0.3413 |
| **LSTM Neural Network** | With Sliding Window | 65.74% | 66.25% | 0.3375 |

### Key Takeaways:
- **The Support Vector Machine (SVM)** working directly on the PCA-reduced feature set yielded the highest classification metrics with **90.13% Test Accuracy**.
- Sequential windowing using only historic target states ($t-1$ to $t-7$) resulted in lower accuracy, indicating that the multi-agent spatial features captured via PCA are critical for accurate swarm state detection.
