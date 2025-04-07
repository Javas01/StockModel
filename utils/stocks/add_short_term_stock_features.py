import numpy as np
import yfinance as yf
import pandas as pd
import talib as ta  # Using TA-Lib instead of pandas_ta

def add_short_term_stock_features(stock_data, ticker):
    # Start with a copy of the data
    df = stock_data.copy()
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        # The levels are (Price_Type, Ticker), so we need to swap and select
        df.columns = df.columns.swaplevel(0, 1)
        df = df.loc[:, ticker]  # Select the ticker level first
    
    # Now we should have simple columns: Open, High, Low, Close, Volume
    print(f"Columns after processing: {df.columns}")
    
    try:
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
        
        # Convert to numpy arrays for TA-Lib
        # Ensure we're working with 1D float arrays
        close_array = df['Close'].astype(float).to_numpy()
        volume_array = df['Volume'].astype(float).to_numpy()
        
        # Technical indicators using TA-Lib
        df['RSI'] = pd.Series(ta.RSI(close_array), index=df.index)
        
        # MACD
        macd, signal, hist = ta.MACD(close_array)
        df['MACD'] = pd.Series(macd, index=df.index)
        df['MACD_signal'] = pd.Series(signal, index=df.index)
        df['MACD_hist'] = pd.Series(hist, index=df.index)
        
        # Bollinger Bands
        df['BBM'] = pd.Series(ta.SMA(close_array, timeperiod=20), index=df.index)
        std = df['Close'].rolling(window=20).std()
        df['BBU'] = df['BBM'] + (std * 2)
        df['BBL'] = df['BBM'] - (std * 2)
        
        # StochRSI
        fastk, fastd = ta.STOCHRSI(close_array)
        df['STOCH_RSI_K'] = pd.Series(fastk, index=df.index)
        df['STOCH_RSI_D'] = pd.Series(fastd, index=df.index)
        
        # Moving averages
        df['EMA_10'] = pd.Series(ta.EMA(close_array, timeperiod=10), index=df.index)
        df['EMA_20'] = pd.Series(ta.EMA(close_array, timeperiod=20), index=df.index)
        ma20 = df['Close'].rolling(window=20).mean()
        ma50 = df['Close'].rolling(window=50).mean()
        df['above_20ma'] = (df['Close'] > ma20).astype(float)
        df['above_50ma'] = (df['Close'] > ma50).astype(float)
        
        # On-Balance Volume
        df['OBV'] = pd.Series(ta.OBV(close_array, volume_array), index=df.index)
        
    except Exception as e:
        print(f"Error calculating indicators for {ticker}: {str(e)}")
        return None
    
    # List of required features
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
    
    # Select only the required features
    result_df = df[required_features]
    
    # Handle NaN values
    result_df = result_df.ffill().bfill()
    
    # Final check for any remaining NaN values
    if result_df.isnull().any().any():
        print(f"Warning: NaN values remain in features for {ticker}")
        result_df = result_df.fillna(0)
    
    return result_df