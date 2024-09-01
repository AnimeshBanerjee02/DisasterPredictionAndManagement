from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

app = Flask(__name__)

# Load earthquake data
data = pd.read_csv('all_month.csv')

# Drop rows with missing values in the target variable (magnitude)
data.dropna(subset=['mag'], inplace=True)

# Extract features (latitude, longitude, date) and target variable (magnitude)
X = data[['latitude', 'longitude', 'time']].copy()
y = data['mag'].values

# Preprocess the time column
X['time'] = X['time'].apply(lambda t: datetime.fromisoformat(t[:-1]).timestamp())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for CNN (assuming each feature is considered as a channel)
X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the CNN model
model_cnn = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])

model_cnn.compile(optimizer='adam', loss='mse')

# Train the CNN model
model_cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=1)

# Training and testing accuracy for CNN
train_loss_cnn = model_cnn.evaluate(X_train_cnn, y_train, verbose=0)
test_loss_cnn = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)

print("Training Loss (CNN):", train_loss_cnn)
print("Testing Loss (CNN):", test_loss_cnn)
