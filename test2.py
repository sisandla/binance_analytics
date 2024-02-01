import ccxt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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

def fetch_historical_data(symbol, timeframe, limit=100):
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

def plot_mean_reversion_strategy(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting closing prices
    ax.plot(data.index, data['close'], label='Close Price', linewidth=2)

    # Plotting Bollinger Bands
    ax.plot(data.index, data['bb_upper'], label='Upper Bollinger Band', linestyle='--', color='red')
    ax.plot(data.index, data['bb_lower'], label='Lower Bollinger Band', linestyle='--', color='green')

    # Highlighting mean reversion points (crossing lower/upper Bollinger Band)
    mean_reversion_points = data[(data['close'] > data['bb_upper']) | (data['close'] < data['bb_lower'])]
    ax.scatter(mean_reversion_points.index, mean_reversion_points['close'], marker='X', color='black', label='Mean Reversion Point')

    ax.set_title('Mean Reversion Strategy with Bollinger Bands')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    st.title('Mean Reversion Trading Strategy with Bollinger Bands, RSI, and MACD')

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

    # Display the first few rows of the data
    st.write('Historical Data with Technical Indicators:')
    st.write(historical_data.head())

    # Plot mean reversion strategy
    st.write('Mean Reversion Strategy with Bollinger Bands:')
    plot_mean_reversion_strategy(historical_data)