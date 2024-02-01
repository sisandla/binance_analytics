import ccxt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
import xml.etree.ElementTree as ET


def read_api_keys_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    api_key = root.find('api_key').text
    api_secret = root.find('api_secret').text

    return api_key, api_secret

# Replace with your Binance API key and secret
# api_key = 'your_api_key'
# api_secret = 'your_api_secret'

# Initialize the Binance exchange object
xml_file_path = './config.xml'
api_key, api_secret = read_api_keys_from_xml(xml_file_path)

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})

def fetch_historical_data(symbol, timeframe, limit=300):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def apply_technical_indicators(data):
    # Bollinger Bands
    indicator_bb = BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_upper'] = indicator_bb.bollinger_hband()
    data['bb_lower'] = indicator_bb.bollinger_lband()

    # RSI (Relative Strength Index)
    indicator_rsi = RSIIndicator(close=data['close'], window=14)
    data['rsi'] = indicator_rsi.rsi()

    # MACD (Moving Average Convergence Divergence)
    indicator_macd = MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = indicator_macd.macd()
    data['macd_signal'] = indicator_macd.macd_signal()

    return data

def create_labels(data, look_forward_period=5):
    # Create binary labels: 1 if the price increases after 'look_forward_period' periods, 0 otherwise
    data['future_price'] = data['close'].shift(-look_forward_period)
    data['label'] = np.where(data['future_price'] > data['close'], 1, 0)
    data.dropna(inplace=True)
    return data

def train_predictive_model(data):
    # Features for the predictive model
    features = ['bb_upper', 'bb_lower', 'rsi', 'macd', 'macd_signal']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['label'], test_size=0.2, random_state=42)

    # Train a simple Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Model Accuracy: {accuracy}")

    return model

def plot_predicted_outcome(data, model):
    # Plotting the closing prices
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['close'], label='Close Price', linewidth=2)

    # Plotting Bollinger Bands
    ax.plot(data.index, data['bb_upper'], label='Upper Bollinger Band', linestyle='--', color='red')
    ax.plot(data.index, data['bb_lower'], label='Lower Bollinger Band', linestyle='--', color='green')

    # Highlighting predicted mean-reversion points
    predicted_mean_reversion = data[model.predict(data[['bb_upper', 'bb_lower', 'rsi', 'macd', 'macd_signal']]) == 0]
    ax.scatter(predicted_mean_reversion.index, predicted_mean_reversion['close'], marker='X', color='black', label='Predicted Mean Reversion Point')

    ax.set_title('Predicted Mean Reversion Strategy with Bollinger Bands, RSI, and MACD')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    st.title('Predicted Mean Reversion Trading Strategy with Bollinger Bands, RSI, and MACD')

    # User input
    symbol = st.selectbox('Select Symbol:', ['BTC/USDT', 'ETH/USDT'])
    timeframe = st.selectbox('Select Timeframe:', ['1h', '4h', '1d'])

    # Fetch historical data
    try:
        historical_data = fetch_historical_data(symbol, timeframe)
    except ccxt.NetworkError as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    # Apply technical indicators
    historical_data = apply_technical_indicators(historical_data)

    # Create labels for the predictive model
    historical_data = create_labels(historical_data)

    # Train predictive model
    model = train_predictive_model(historical_data)

    # Display the first few rows of the data with labels
    st.write('Historical Data with Labels:')
    st.write(historical_data.head())

    # Plot predicted mean reversion strategy
    st.write('Predicted Mean Reversion Strategy:')
    plot_predicted_outcome(historical_data, model)