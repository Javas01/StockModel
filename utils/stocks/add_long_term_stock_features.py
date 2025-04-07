import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta  # Using TA-Lib instead of pandas_ta

def add_long_term_stock_features(stock_data, ticker):
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
        df['20d_vol'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Bollinger Bands
        df['BBM'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BBU'] = df['BBM'] + (std * 2)
        df['BBL'] = df['BBM'] - (std * 2)
        
        # Moving average
        ma50 = df['Close'].rolling(window=50).mean()
        df['above_50ma'] = (df['Close'] > ma50).astype(float)
        
        # Get fundamental data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            df['P_B_Ratio'] = info.get('priceToBook', np.nan)
        except Exception as e:
            print(f"Error getting P/B ratio for {ticker}: {e}")
            df['P_B_Ratio'] = np.nan
            
    except Exception as e:
        print(f"Error calculating indicators for {ticker}: {str(e)}")
        return None
    
    # List of required features
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '1w_return', '2w_return', '1m_return',
        '20d_vol',
        'BBL', 'BBM', 'BBU',
        'above_50ma',
        'P_B_Ratio'
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