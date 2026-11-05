
🌧️ Rain Prediction Using Artificial Neural Network (ANN)

📌 Project Overview

This project focuses on predicting whether it will rain tomorrow using an Artificial Neural Network (ANN). The model is trained on historical Australian weather data to analyze weather conditions and predict rainfall for the next day.

The project uses data preprocessing, data visualization, feature scaling, and deep learning techniques to build a binary classification model.

🎯 Project Objectives

Predict whether it will rain tomorrow.

Analyze historical weather conditions.

Perform data cleaning and preprocessing.

Build an Artificial Neural Network (ANN) model.

Evaluate model performance using classification metrics.

📊 Dataset

Dataset: Australian Weather Dataset (weatherAUS.csv)

The dataset contains approximately 10 years of daily weather observations from different locations across Australia.

Target Variable: RainTomorrow

Yes – Rain is expected tomorrow.

No – Rain is not expected tomorrow.

🛠️ Technologies & Libraries Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

TensorFlow / Keras

Joblib

⚙️ Project Workflow

Data Loading – Import the Australian weather dataset.

Data Visualization – Analyze rainfall distribution and correlations.

Data Cleaning – Handle missing values using mode and median.

Feature Engineering – Apply cyclic encoding to month and day.

Data Preprocessing – Perform label encoding and standard scaling.

Model Building – Develop an ANN using Keras Sequential API.

Model Training – Train the ANN with early stopping.

Model Evaluation – Evaluate predictions using a confusion matrix and classification report.

🧠 ANN Model Architecture

The Artificial Neural Network consists of:

Input Layer – 26 input features

Hidden Layer 1 – 32 neurons (ReLU)

Hidden Layer 2 – 32 neurons (ReLU)

Hidden Layer 3 – 16 neurons (ReLU)

Dropout Layer – 0.25

Hidden Layer 4 – 8 neurons (ReLU)

Dropout Layer – 0.5

Output Layer – 1 neuron (Sigmoid)

Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation Metric: Accuracy

Batch Size: 32

Maximum Epochs: 150 (with Early Stopping)

📈 Model Evaluation

The model is evaluated using:

Accuracy Score

Precision

Recall

F1-Score

Confusion Matrix

Classification Report
