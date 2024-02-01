import streamlit as st
import pandas as pd
import requests
import json
import os
import cryptocompare

from binance.client import Client
# from binance.spot import Spot as Client
import xml.etree.ElementTree as ET
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Analytics",
        page_icon="📈",
    )

    st.write("# Welcome! 👋")
    st.sidebar.success("Selected binance.")

    st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Dataframes & Graphs
    """
    )
    base_url = 'https://api1.binance.com'
    api_call = '/api/v3/ticker/tradingDay'
    headers = {'content-type': 'application/json', 'X-MBX-APIKEY': api_key}
    
    response = requests.get(base_url + api_call, headers=headers)
    response = json.loads(response.text)
    print(response)
    df = pd.DataFrame.from_records(response, index=[1])
    df.head()
    st.dataframe(df)

def read_api_keys_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    api_key = root.find('api_key').text
    api_secret = root.find('api_secret').text

    return api_key, api_secret


def get_binance_tickers(api_key, api_secret):
    # Initialize the Binance client
    client = Client(api_key, api_secret)

    # Fetch all tickers using the client
    tickers_data = client.get_all_tickers()

    # Create a Pandas DataFrame from the tickers data
    tickers_df = pd.DataFrame(tickers_data)

    return tickers_df

def display_table_and_graph(data):
    # Display the Pandas DataFrame
    st.table(data)

    # Display line graph using Streamlit
    st.line_chart(data['price'].astype(float))

def display_line_chart(data):
    st.line_chart(data)
    
    
def fetch_crypto_prices(symbols, start_date, end_date):
    crypto_prices = pd.DataFrame()

    for symbol in symbols:
        prices = cryptocompare.get_historical_price_day(symbol, currency='USD', toTs=end_date, limit=365)
        prices_df = pd.DataFrame(prices)
        crypto_prices[symbol] = prices_df['close']

    crypto_prices.set_index('time', inplace=True)
    return crypto_prices

def get_binance_historical_prices(api_key, api_secret, symbol, interval='1d', limit=365):
    # Initialize the Binance client
    client = Client(api_key, api_secret)

    # Fetch historical prices using the client
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    # Extract and format the data
    prices_data = {
        'time': [pd.to_datetime(x[0], unit='ms') for x in klines],
        'close': [float(x[4]) for x in klines]
    }

    # Create a Pandas DataFrame from the prices data
    prices_df = pd.DataFrame(prices_data)
    prices_df.set_index('time', inplace=True)

    return prices_df

def display_line_chart(data):
    st.line_chart(data)


def main():
    xml_file_path = './config.xml'
    api_key, api_secret = read_api_keys_from_xml(xml_file_path)
    
    st.title('Binance Cryptocurrency Price Correlation')

    binance_crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'USDTUSDT', 'USDCUSDT', 'ALGOUSDT']

    # Fetch historical prices for each cryptocurrency
    prices_data = {}
    for symbol in binance_crypto_symbols:
        prices_data[symbol] = get_binance_historical_prices(api_key, api_secret, symbol)

    # Display line chart
    display_line_chart(pd.DataFrame(prices_data))
    
    # Replace 'YOUR_API_KEY' and 'YOUR_API_SECRET' with your Binance API key and secret
    st.title('Binance Crypto Tickers')

    # Get tickers data
    tickers_df = get_binance_tickers(api_key, api_secret)

    # Display table and graph
    display_table_and_graph(tickers_df)
    
if __name__ == "__main__":
    main()
    # client = Client(api_key, secret_key, testnet=True)
    # tickers = client.get_all_tickers()
    # run()