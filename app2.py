import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

from keras.models import load_model
import streamlit as st
import os

# Define the path to your model file in the repository
model_path = os.path.join(os.getcwd(), 'Stock Predictions Model 2025.keras')

# Load the pre-trained model
model = load_model(model_path)


# Streamlit setup
st.header('Stock Price Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'TCS.NS')

# Define the date range for prediction
today = datetime.now().strftime('%Y-%m-%d')
# Define the date range for prediction next 5 days
end_date = (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d')
next_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 6)]


# Download historical stock data
data = yf.download(stock, start='2020-01-01', end=end_date)

# Display overall data
st.subheader('Overall Data for {}:'.format(stock))
st.write(data)

# Preprocess the data
data_close = data['Close'].values.reshape(-1, 1)  # Extract the 'Close' prices and reshape
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_close)

# Prepare input data for prediction
x_input = []
for i in range(100, len(data_scaled)):
    x_input.append(data_scaled[i-100:i, 0])  # Using the previous 100 data points as features

x_input = np.array(x_input)
x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))  # Reshape for LSTM model

# Make predictions for next 5 days
predicted_prices_scaled = model.predict(x_input[-5:])  # Predict for the last 5 sequences
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

# Display the predicted prices for the next 5 days
st.subheader('Predicted Prices for the Next 5 Days:')
for i, date in enumerate(next_dates):
    st.write('Predicted Price for {}: ₹{:.2f}'.format(date, predicted_prices[i][0]))


# Visualize the actual and predicted prices
plt.figure(figsize=(10, 6))

# Determine color based on price movement
colors = ['red' if predicted_prices[i] < predicted_prices[i-1] else 'green' for i in range(1, len(predicted_prices))]

# Plot the predicted prices
for i in range(1, len(predicted_prices)):
    plt.plot([next_dates[i-1], next_dates[i]], [predicted_prices[i-1], predicted_prices[i]], color=colors[i-1])

plt.xlabel('Dates')
plt.ylabel('Price')
plt.title('Predicted Stock Prices for the Next 5 Days')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
st.pyplot(plt)





