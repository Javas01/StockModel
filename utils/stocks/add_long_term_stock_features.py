import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta  # Using TA-Lib instead of pandas_ta
def add_long_term_stock_features(stock_data, ticker):
    # Start with a copy of the data and ensure we have a clean DataFrame structure
    df = stock_data.copy()
    
    # If we have a multi-index DataFrame, flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level='Ticker', drop=True)
    
    # If Close is a DataFrame, convert it to a Series
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]
    
    # Basic price data - ensure they're Series, not DataFrames
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    
    # Calculate returns
    df['1w_return'] = df['Close'].pct_change(periods=5)  # 5 trading days
    df['2w_return'] = df['Close'].pct_change(periods=10)  # 10 trading days
    df['1m_return'] = df['Close'].pct_change(periods=21)  # 21 trading days
    
    # Calculate volatility
    df['20d_vol'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Fix Bollinger Bands calculation
    try:
        # Ensure Close is a Series
        close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        
        # Calculate Bollinger Bands
        sma = close_series.rolling(window=20).mean()
        std = close_series.rolling(window=20).std()
        
        df['BBM'] = sma
        df['BBU'] = sma + (std * 2)
        df['BBL'] = sma - (std * 2)
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        df['BBM'] = np.nan
        df['BBU'] = np.nan
        df['BBL'] = np.nan
    
    # Calculate moving average
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    ma50 = close_series.rolling(window=50).mean()
    df['above_50ma'] = (close_series > ma50).astype(float)
    
    # Get fundamental data
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        df['P_B_Ratio'] = info.get('priceToBook', np.nan)
    except Exception as e:
        print(f"Error getting P/B ratio for {ticker}: {e}")
        df['P_B_Ratio'] = np.nan
    
    # List of required features
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '1w_return', '2w_return', '1m_return',
        '20d_vol',
        'BBL', 'BBM', 'BBU',
        'above_50ma',
        'P_B_Ratio'
    ]
    
    # Verify all features exist and are Series (not DataFrames)
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