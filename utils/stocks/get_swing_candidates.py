import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def get_swing_candidates(min_volume=1000000, min_option_interest=1000, max_stocks=20):
    """
    Find stocks with high options activity and volatility potential for swing trading.
    """
    print("\n=== Options Swing Trading Screening Started ===")
    
    # Base universe - Include major ETFs and high-volume stocks
    base_universe = {
        'ETFs': ['SPY', 'QQQ', 'IWM', 'EEM', 'XLF', 'XLE', 'XLK', 'ARKK'],  # Major ETFs
        'Tech': ['AAPL', 'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL', 'MSFT'],    # Tech leaders
        'Momentum': ['PLTR', 'LCID', 'RIVN', 'COIN', 'RBLX', 'U', 'NET'],    # High-beta stocks
        'Meme': ['GME', 'AMC', 'BBBY', 'BB', 'NOK'],                         # High social sentiment
    }
    
    all_tickers = [ticker for category in base_universe.values() for ticker in category]
    print(f"Analyzing {len(all_tickers)} potential swing trading candidates")
    
    swing_candidates = []
    
    # Analysis window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days for recent activity
    
    for ticker in all_tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                continue
                
            # Calculate volatility metrics
            hist['Returns'] = hist['Close'].pct_change()
            daily_vol = hist['Returns'].std() * np.sqrt(252)  # Annualized volatility
            
            # Get options data
            try:
                expirations = stock.options
                if not expirations:
                    continue
                
                # Analyze front-month and next-month options
                near_exp = expirations[0]
                next_exp = expirations[1] if len(expirations) > 1 else near_exp
                
                # Get options chains
                near_chain = stock.option_chain(near_exp)
                next_chain = stock.option_chain(next_exp)
                
                # Calculate options metrics
                current_price = hist['Close'].iloc[-1]
                
                # Near-term ATM options
                near_atm_calls = near_chain.calls[
                    (near_chain.calls['strike'] - current_price).abs() <= current_price * 0.05]
                near_atm_puts = near_chain.puts[
                    (near_chain.puts['strike'] - current_price).abs() <= current_price * 0.05]
                
                # Options activity metrics
                total_oi = (near_chain.calls['openInterest'].sum() + 
                          near_chain.puts['openInterest'].sum())
                total_volume = (near_chain.calls['volume'].sum() + 
                              near_chain.puts['volume'].sum())
                
                # Implied Volatility analysis
                near_atm_iv = (near_atm_calls['impliedVolatility'].mean() + 
                              near_atm_puts['impliedVolatility'].mean()) / 2
                
                # Volume spike detection
                avg_volume = hist['Volume'].mean()
                latest_volume = hist['Volume'].iloc[-1]
                volume_spike = latest_volume / avg_volume
                
                # Swing potential score (custom metric)
                swing_score = (
                    daily_vol * 0.4 +                    # Weight for historical volatility
                    (near_atm_iv / 2) * 0.3 +           # Weight for implied volatility
                    (volume_spike / 10) * 0.2 +         # Weight for volume activity
                    (total_volume / total_oi) * 0.1     # Weight for options activity
                )
                
                swing_candidates.append({
                    'ticker': ticker,
                    'price': current_price,
                    'daily_volatility': daily_vol * 100,  # Convert to percentage
                    'implied_volatility': near_atm_iv * 100,
                    'volume_spike': volume_spike,
                    'options_volume': total_volume,
                    'open_interest': total_oi,
                    'swing_score': swing_score,
                    'avg_option_spread': (
                        near_atm_calls['ask'].mean() - near_atm_calls['bid'].mean()
                    ),
                    'category': next(
                        cat for cat, tickers in base_universe.items() 
                        if ticker in tickers
                    )
                })
                
            except Exception as e:
                print(f"Error processing options data for {ticker}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    # Sort candidates by swing score
    swing_candidates.sort(key=lambda x: x['swing_score'], reverse=True)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(swing_candidates)
    if not df.empty:
        # Format for display
        df['daily_volatility'] = df['daily_volatility'].round(2).astype(str) + '%'
        df['implied_volatility'] = df['implied_volatility'].round(2).astype(str) + '%'
        df['volume_spike'] = df['volume_spike'].round(2).astype(str) + 'x'
        df['swing_score'] = df['swing_score'].round(2)
        df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
        
        print("\n=== Top Swing Trading Candidates ===")
        print(f"Total candidates analyzed: {len(all_tickers)}")
        print(f"Candidates meeting criteria: {len(swing_candidates)}")
        print("\nTop Candidates by Swing Score:")
        print(df.head(max_stocks).to_string(index=False))
        
        # Group analysis
        print("\nCategory Analysis:")
        category_stats = df.groupby('category')['swing_score'].agg(['mean', 'count'])
        print(category_stats)
        
    return [stock['ticker'] for stock in swing_candidates[:max_stocks]]