import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

# Load dataset
file_path = "apple_stock.xlsx"  # Adjust if necessary
data = pd.read_excel(file_path)

# Use the 'Close' price for prediction
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize data (LSTMs work better with scaled data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Define sequence length
seq_length = 25  

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Prepare sequences
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape X_train and X_test for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
print("Building the LSTM model with sequence length:", seq_length)
model = Sequential([
    Input(shape=(seq_length, 1)),  # Using Input() layer to define shape
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
print("Model compilation complete.")

# Train the model
print("Starting model training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
print("Model training complete!")

# Save the model in the recommended Keras format
model.save("lstm_stock_model.keras")
print("Model saved as 'lstm_stock_model.keras'. Training process finished successfully!")

