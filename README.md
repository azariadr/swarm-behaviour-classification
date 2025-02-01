# Swarm Behaviour Classification

## Overview

This project aims to classify swarm behavior using machine learning algorithms. The dataset used contains data on the position, velocity, and acceleration of each agent in the swarm.

## Dataset

The dataset used is "Swarm_Behaviour.csv". This dataset contains 2401 features representing the position, velocity, acceleration, and other information of each agent in the swarm. The target feature is "Swarm_Behaviour", which indicates the behavior of the swarm.

## Preprocessing

The data is preprocessed using StandardScaler to scale the features and PCA to reduce dimensionality. PCA is used to reduce the number of features from 2401 to 400.

## Model

Several machine learning models are used to classify swarm behavior, including:

* Naive Bayes
* Hidden Markov Model
* SVM
* LSTM

## Results

The results of each model are evaluated using accuracy and mean absolute error. The results show that the LSTM model with a sliding window provides the best results.

## Usage

To run this code, you need to upload the "Swarm_Behaviour.csv" dataset to your Google Drive. Then, you can run the code in Google Colab.
