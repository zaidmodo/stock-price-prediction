import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fetch_live_stock_data(ticker, period='60d', interval='1h'):
    """Fetch real-time stock data from Yahoo Finance."""
    print(f"📡 Fetching live stock data for {ticker}...")
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

# Load trained model
print("\n🔄 Loading trained model...")
model = load_model("lstm_stock_model.keras")
print("✔️ Model loaded successfully!")

# Fetch live data
ticker = input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOG): ")
live_data = fetch_live_stock_data(ticker)

# Extract closing prices
closing_prices = live_data['Close'].values.reshape(-1, 1)
open_prices = live_data['Open'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Define sequence length (same as used in training)
seq_length = 25

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
    return np.array(X)

# Prepare sequences
X_live = create_sequences(scaled_data, seq_length)
X_live = X_live.reshape((X_live.shape[0], X_live.shape[1], 1))

# Make predictions
print("\n📊 Predicting future stock prices...")
predictions = model.predict(X_live)
predicted_prices = scaler.inverse_transform(predictions)

# Extract recent actual prices
actual_prices = closing_prices[-len(predicted_prices):]
open_prices_recent = open_prices[-len(predicted_prices):]

# Calculate error metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
accuracy = 100 - mape

# Print performance metrics
print("\n📊✨ Real-Time Model Performance ✨📊")
print(f"📉 MAE:       ${mae:.2f}")
print(f"📉 RMSE:      ${rmse:.2f}")
print(f"📈 MAPE:      {mape:.2f}%")
print(f"🎯 Accuracy:  {accuracy:.2f}%")

# Plot Predictions vs Actual Prices
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
plt.plot(actual_prices, color='blue', linewidth=2, label='Actual Prices')
plt.plot(predicted_prices, color='red', linestyle='dashed', linewidth=2, label='Predicted Prices')
plt.plot(open_prices_recent, color='green', linestyle='dotted', linewidth=2, label='Open Prices')
plt.title(f'📈 {ticker} Stock Prediction (Accuracy: {accuracy:.2f}%)', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.show()

# Show Pie Chart for Accuracy
tags = ['Accuracy', 'Error']
values = [accuracy, 100 - accuracy]
plt.figure(figsize=(6, 6))
plt.pie(values, labels=tags, autopct='%1.1f%%', colors=['green', 'red'], startangle=140)
plt.title("📊 Accuracy Distribution")
plt.show()
