import pandas as pd
import yfinance as yf

def get_high_interest_tickers(min_volume=1000000, min_option_interest=1000, max_stocks=20):
    """
    Find stocks with high trading volume and significant options interest, focusing on mid-cap stocks with higher volatility.
    """
    print("\n=== Stock Screening Started ===")
    
    stocks_picks_from_chatgpt = ["TSLA", "AMZN", "NVDA", "GOOGL", "MSFT", "META", "AAPL", "AMD", "SHOP", "SQ", "PYPL", "BABA", "SPGI", "V", "MA", "INTC", "ZM", "SNAP", "DOCU", "TWLO", "CRWD", "ROKU", "UBER", "ADBE", "INTU", "DOCU"]
    
    high_growth_stocks = ["TSLA", "AMZN", "NVDA", "GOOGL", "MSFT", "META", "AAPL", "AMD", "SHOP", "SQ", "PYPL", "BABA", "SPGI", "V", "MA", "INTC", "ZM", "SNAP", "DOCU", "TWLO", "CRWD", "ROKU", "UBER", "ADBE", "INTU", "DOCU"]

    potential_rally_stocks = ["LULU", "OLLI", "CAG", "GIS", "LW", "PAYX", "BEKE", "SBUX", "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "TSLA", "META", "BABA", "PYPL", "ADBE", "AMD", "SPGI", "INTC", "CRM", "PEP", "KO", "MCD", "WMT", "COST", "TJX", "LOW", "HD", "DIS", "V", "MA", "XOM", "CVX"]

    # Combine all sectors and remove duplicates
    all_tickers = list(set(stocks_picks_from_chatgpt + high_growth_stocks + potential_rally_stocks))
    
    print(f"Analyzing {len(all_tickers)} growth and momentum stocks")
    print(f"Minimum daily volume: {min_volume:,}")
    print(f"Minimum option interest: {min_option_interest:,}")
    
    print(f"Found {len(all_tickers)} unique tickers to analyze")
    print(f"Minimum daily volume: {min_volume:,}")
    print(f"Minimum option interest: {min_option_interest:,}")

    high_interest_stocks = []
    stocks_processed = 0
    errors = 0
    
    for i, ticker in enumerate(all_tickers, 1):
        try:
            print(f"Processing {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get options expiration dates
            try:
                expirations = stock.options
                if not expirations:
                    print(f"No options data available for {ticker}")
                    continue
                    
                # Get the first expiration date
                first_expiration = expirations[0]
                options = stock.option_chain(first_expiration)
                total_oi = options.calls['openInterest'].sum() + options.puts['openInterest'].sum()
                put_call_ratio = options.puts['openInterest'].sum() / max(1, options.calls['openInterest'].sum())
                
                print(f"{ticker} options expiration: {first_expiration}, Total OI: {total_oi}")
                
            except Exception as e:
                print(f"Error getting options data for {ticker}: {str(e)}")
                continue
            
            # Check volume and options interest
            volume = info.get('averageVolume', 0)
            if volume > min_volume and total_oi > min_option_interest:
                high_interest_stocks.append({
                    'ticker': ticker,
                    'name': info.get('shortName', 'N/A'),
                    'price': info.get('currentPrice', 0),
                    'volume': volume,
                    'options_oi': total_oi,
                    'put_call_ratio': put_call_ratio,
                    'expiration': first_expiration,  # Added expiration date
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 0)
                })
                print(f"Added {ticker} to high interest stocks")
            
            stocks_processed += 1
                
        except Exception as e:
            errors += 1
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Sort by options open interest
    high_interest_stocks.sort(key=lambda x: x['options_oi'], reverse=True)
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame(high_interest_stocks)
    if not df.empty:
        # Format the columns for display
        df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
        df['volume'] = df['volume'].apply(lambda x: f"{x/1e6:.1f}M")
        df['options_oi'] = df['options_oi'].apply(lambda x: f"{x/1000:.1f}K")
        df['put_call_ratio'] = df['put_call_ratio'].apply(lambda x: f"{x:.2f}")
        df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
        df['beta'] = df['beta'].apply(lambda x: f"{x:.2f}")
        
        print("\n=== Results ===")
        print(f"Total stocks scanned: {stocks_processed}")
        print(f"Stocks meeting criteria: {len(high_interest_stocks)}")
        print(f"Errors encountered: {errors}")
        print("\nTop Stocks by Options Interest:")
        print(df.head(max_stocks).to_string(index=False))
    else:
        print("\nNo stocks met the criteria. Try adjusting the minimum requirements.")
    
    return [stock['ticker'] for stock in high_interest_stocks[:max_stocks]]