import numpy as np
import yfinance as yf
import pandas as pd
import talib as ta  # Using TA-Lib instead of pandas_ta
import constants

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

           # Add options-specific indicators
        try:
            stock = yf.Ticker(ticker)
            
            # Get current price from the most recent close
            current_price = df['Close'].iloc[-1]
            
            # Get options data for the nearest expiration
            expirations = stock.options
            if expirations:
                nearest_expiry = expirations[0]
                options_chain = stock.option_chain(nearest_expiry)
                
                # Calculate Put-Call Ratio
                total_call_oi = options_chain.calls['openInterest'].sum()
                total_put_oi = options_chain.puts['openInterest'].sum()
                df['Put_Call_Ratio'] = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
                
                # Calculate Options Volume
                df['Call_Volume'] = options_chain.calls['volume'].sum()
                df['Put_Volume'] = options_chain.puts['volume'].sum()
                df['Options_Volume'] = df['Call_Volume'] + df['Put_Volume']
                
                # Calculate Open Interest
                df['Call_OI'] = total_call_oi
                df['Put_OI'] = total_put_oi
                
                # Calculate IV Percentile (using front-month ATM options)
                # Find ATM options (within 5% of current price)
                atm_calls = options_chain.calls[
                    (options_chain.calls['strike'] >= current_price * 0.95) & 
                    (options_chain.calls['strike'] <= current_price * 1.05)
                ]
                atm_puts = options_chain.puts[
                    (options_chain.puts['strike'] >= current_price * 0.95) & 
                    (options_chain.puts['strike'] <= current_price * 1.05)
                ]
                
                # Get average IV from ATM options
                if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                    calls_iv = atm_calls['impliedVolatility'].mean()
                else:
                    calls_iv = 0.5
                    
                if not atm_puts.empty and 'impliedVolatility' in atm_puts.columns:
                    puts_iv = atm_puts['impliedVolatility'].mean()
                else:
                    puts_iv = 0.5
                
                current_iv = (calls_iv + puts_iv) / 2
                
                # Fill the DataFrame with the calculated IV
                df['IV_Percentile'] = 50  # Default value
                
                # Calculate IV percentile if we have enough history
                if len(df) >= 20:  # At least 20 days of data
                    historical_iv = [current_iv] * len(df)  # Simple proxy
                    iv_percentile = (sum(iv < current_iv for iv in historical_iv[-252:]) / 
                                   min(len(historical_iv), 252) * 100)
                    df['IV_Percentile'] = iv_percentile
                
            else:
                # Set default values if no options data available
                df['Put_Call_Ratio'] = 1.0
                df['IV_Percentile'] = 50
                df['Options_Volume'] = 0
                df['Call_Volume'] = 0
                df['Put_Volume'] = 0
                df['Call_OI'] = 0
                df['Put_OI'] = 0
                
        except Exception as e:
            print(f"Error calculating options indicators for {ticker}: {e}")
            df['Put_Call_Ratio'] = 1.0
            df['IV_Percentile'] = 50
            df['Options_Volume'] = 0
            df['Call_Volume'] = 0
            df['Put_Volume'] = 0
            df['Call_OI'] = 0
            df['Put_OI'] = 0
        
    except Exception as e:
        print(f"Error calculating indicators for {ticker}: {str(e)}")
        return None
    
    # List of required features
    required_features = constants.SHORT_TERM_STOCK_FEATURES
    
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