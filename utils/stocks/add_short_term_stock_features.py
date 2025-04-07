import numpy as np
import yfinance as yf
import pandas as pd
import talib as ta  # Using TA-Lib instead of pandas_ta

def add_short_term_stock_features(stock_data, ticker):
    # Start with a copy of the data and ensure we have a clean DataFrame structure
    df = stock_data.copy()
    
    # If we have a multi-index DataFrame, flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level='Ticker', drop=True)
    
    # Convert DataFrame columns to Series if needed
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    
    # Calculate returns
    df['1w_return'] = df['Close'].pct_change(periods=5)
    df['2w_return'] = df['Close'].pct_change(periods=10)
    df['1m_return'] = df['Close'].pct_change(periods=21)
    
    # Calculate volatility
    df['10d_vol'] = df['Close'].pct_change().rolling(window=10).std()
    df['20d_vol'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Technical indicators using TA-Lib
    # Convert to numpy array for TA-Lib
    close_array = df['Close'].astype(float).values
    volume_array = df['Volume'].astype(float).values
    
    # RSI
    df['RSI'] = ta.RSI(close_array)
    
    # MACD
    macd, signal, hist = ta.MACD(close_array)
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    
    # Bollinger Bands
    df['BBM'] = ta.SMA(close_array, timeperiod=20)
    std = df['Close'].rolling(window=20).std()
    df['BBU'] = df['BBM'] + (std * 2)
    df['BBL'] = df['BBM'] - (std * 2)
    
    # StochRSI
    fastk, fastd = ta.STOCHRSI(close_array)
    df['STOCH_RSI_K'] = fastk
    df['STOCH_RSI_D'] = fastd
    
    # Moving averages
    df['EMA_10'] = ta.EMA(close_array, timeperiod=10)
    df['EMA_20'] = ta.EMA(close_array, timeperiod=20)
    ma20 = df['Close'].rolling(window=20).mean()
    ma50 = df['Close'].rolling(window=50).mean()
    df['above_20ma'] = (df['Close'] > ma20).astype(float)
    df['above_50ma'] = (df['Close'] > ma50).astype(float)
    
    # On-Balance Volume
    df['OBV'] = ta.OBV(close_array, volume_array)
    
    # List of required features from get_short_term_stock_features()
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '1w_return', '2w_return', '1m_return',
        '10d_vol', '20d_vol',
        'Volume_SMA_20', 'Volume_SMA_50', 'Volume_Change',
        'STOCH_RSI_K', 'STOCH_RSI_D',
        'RSI',
        'MACD', 'MACD_signal', 'MACD_hist',
        'BBL', 'BBM', 'BBU',
        'above_50ma', 'above_20ma',
        'OBV',
        'EMA_10', 'EMA_20'
    ]
    
    # Verify all features exist
    for feature in required_features:
        if feature not in df.columns:
            print(f"Missing feature: {feature}")
            df[feature] = np.nan
        elif isinstance(df[feature], pd.DataFrame):
            df[feature] = df[feature].iloc[:, 0]
    
    # Select only the required features and handle NaN values
    result_df = df[required_features].ffill().bfill()
    
    # Ensure all columns are numeric
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    return result_df