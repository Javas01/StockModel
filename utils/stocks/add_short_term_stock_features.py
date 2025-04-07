import numpy as np
import yfinance as yf
import pandas as pd
import talib as ta  # Using TA-Lib instead of pandas_ta

def add_short_term_stock_features(stock_data, ticker):
    # Get options data
    stock = yf.Ticker(ticker)
    try:
        opt = stock.option_chain('nearest')
        has_options = True
    except:
        has_options = False
    
    # Basic price momentum indicators
    df = stock_data.copy()
    
    # Short-term returns (1-week, 2-week, 1-month)
    df['1w_return'] = df['Close'].pct_change(periods=5)  # 5 trading days in a week
    df['2w_return'] = df['Close'].pct_change(periods=10)  # 10 trading days in 2 weeks
    df['1m_return'] = df['Close'].pct_change(periods=21)  # 21 trading days in 1 month
    
    # Short-term volatility (10-day, 20-day rolling volatility)
    df['10d_vol'] = df['Close'].rolling(window=10).std()
    df['20d_vol'] = df['Close'].rolling(window=20).std()
    
    # Volume-related indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'] / df['Volume_SMA_20']  # Volume relative to 20-day average
    
    # Technical indicators using TA-Lib
    close_prices = df['Close'].to_numpy().flatten()  # Ensure it is 1D

    # Add Stochastic RSI (returns fastk and fastd)
    fastk, fastd = ta.STOCHRSI(close_prices, timeperiod=14)
    df['STOCH_RSI_K'] = pd.Series(fastk, index=df.index)
    df['STOCH_RSI_D'] = pd.Series(fastd, index=df.index)

    # RSI (Relative Strength Index)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)
    
    # MACD and related indicators
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    df['BBL'], df['BBM'], df['BBU'] = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    # Momentum and Trend-following Indicators
    df['above_50ma'] = df['Close'] > df['Close'].rolling(window=50).mean()  # Price above 50-day MA
    df['above_20ma'] = df['Close'] > df['Close'].rolling(window=20).mean()  # Price above 20-day MA
    
    # On-Balance Volume (OBV)
    df['OBV'] = ta.OBV(close_prices, df['Volume'].to_numpy())
    
    # Add Exponential Moving Averages (EMA) for short-term trends
    df['EMA_10'] = ta.EMA(close_prices, timeperiod=10)
    df['EMA_20'] = ta.EMA(close_prices, timeperiod=20)
    
    # Options-related indicators (if available)
    if has_options:
        df['call_oi'] = opt.calls['openInterest'].sum()
        df['put_call_ratio'] = opt.puts['openInterest'].sum() / opt.calls['openInterest'].sum()
        df['call_volume'] = opt.calls['volume'].sum()
    
    # Return the dataframe with the indicators
    return df
