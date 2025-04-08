import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def get_breakout_stocks(min_market_cap=1e9, max_stocks=35):
    """
    Identify stocks with strong potential for 2-5 year growth based on fundamental and technical factors.
    """
    print("\n=== Long-term Growth Stock Screening Started ===")
    
    # Base universe of stocks to analyze
    base_universe = {
        'Tech_Leaders': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'Growth_Tech': ['AMD', 'ADBE', 'CRM', 'SHOP', 'SNOW', 'NET', 'CRWD', 'DDOG'],
        'FinTech': ['V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD'],
        'Healthcare': ['UNH', 'JNJ', 'PFE', 'MRNA', 'ISRG', 'VEEV'],
        'Consumer': ['LULU', 'SBUX', 'NKE', 'COST', 'TGT', 'HD'],
        'Industrial': ['CAT', 'DE', 'HON', 'UNP', 'GE'],
        'Clean_Energy': ['ENPH', 'SEDG', 'RUN', 'STEM', 'FSLR']
    }
    
    all_tickers = [ticker for category in base_universe.values() for ticker in category]
    print(f"Analyzing {len(all_tickers)} potential growth candidates")
    
    growth_candidates = []
    
    # Analysis windows
    end_date = datetime.now()
    start_date_5y = end_date - timedelta(days=365*5)
    start_date_1y = end_date - timedelta(days=365)
    
    for ticker in all_tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical data
            hist_5y = stock.history(start=start_date_5y, end=end_date)
            hist_1y = hist_5y.last('365D')
            
            if hist_5y.empty:
                continue
            
            # Market cap filter
            market_cap = info.get('marketCap', 0)
            if market_cap < min_market_cap:
                continue
            
            # Calculate growth metrics
            try:
                # Financial health metrics
                revenue_growth = info.get('revenueGrowth', 0)
                profit_margin = info.get('profitMargins', 0)
                debt_to_equity = info.get('debtToEquity', 1000)
                roa = info.get('returnOnAssets', 0)
                roe = info.get('returnOnEquity', 0)
                
                # Valuation metrics
                forward_pe = info.get('forwardPE', 100)
                pb_ratio = info.get('priceToBook', 10)
                
                # Technical indicators
                hist_5y['SMA200'] = hist_5y['Close'].rolling(window=200).mean()
                hist_5y['SMA50'] = hist_5y['Close'].rolling(window=50).mean()
                
                current_price = hist_5y['Close'].iloc[-1]
                sma200 = hist_5y['SMA200'].iloc[-1]
                sma50 = hist_5y['SMA50'].iloc[-1]
                
                # Calculate volatility and returns
                returns_1y = hist_1y['Close'].pct_change()
                volatility_1y = returns_1y.std() * np.sqrt(252)
                total_return_5y = (current_price / hist_5y['Close'].iloc[0]) - 1
                
                # R&D and Innovation (if available)
                rd_to_revenue = info.get('researchAndDevelopmentToRevenue', 0)
                
                # Institutional ownership
                inst_ownership = info.get('institutionalOwnership', 0)
                
                # Growth score calculation
                growth_score = (
                    (revenue_growth * 0.20) +                    # Revenue growth
                    (profit_margin * 0.15) +                     # Profitability
                    (min(roe, 1) * 0.15) +                      # Return on equity
                    (min(roa, 1) * 0.10) +                      # Return on assets
                    (rd_to_revenue * 0.10) +                    # R&D investment
                    (inst_ownership * 0.10) +                   # Institutional confidence
                    ((current_price > sma200) * 0.10) +        # Long-term trend
                    ((current_price > sma50) * 0.10)           # Short-term trend
                )
                
                # Risk adjustment
                risk_factor = (
                    (min(debt_to_equity / 100, 1) * -0.3) +    # Penalize high debt
                    (volatility_1y * -0.2) +                   # Penalize high volatility
                    (min(forward_pe / 50, 1) * -0.3) +        # Penalize high valuation
                    (min(pb_ratio / 10, 1) * -0.2)           # Penalize high P/B ratio
                )
                
                # Final score
                final_score = growth_score + risk_factor
                
                growth_candidates.append({
                    'ticker': ticker,
                    'category': next(cat for cat, tickers in base_universe.items() if ticker in tickers),
                    'market_cap': market_cap,
                    'current_price': current_price,
                    'revenue_growth': revenue_growth * 100 if revenue_growth else 0,
                    'profit_margin': profit_margin * 100 if profit_margin else 0,
                    'roe': roe * 100 if roe else 0,
                    'forward_pe': forward_pe,
                    '5y_return': total_return_5y * 100,
                    'inst_ownership': inst_ownership * 100 if inst_ownership else 0,
                    'growth_score': growth_score,
                    'risk_factor': risk_factor,
                    'final_score': final_score
                })
                
            except Exception as e:
                print(f"Error calculating metrics for {ticker}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    # Sort candidates by final score
    growth_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(growth_candidates)
    if not df.empty:
        # Format for display
        df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
        df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
        df['revenue_growth'] = df['revenue_growth'].round(2).astype(str) + '%'
        df['profit_margin'] = df['profit_margin'].round(2).astype(str) + '%'
        df['roe'] = df['roe'].round(2).astype(str) + '%'
        df['5y_return'] = df['5y_return'].round(2).astype(str) + '%'
        df['inst_ownership'] = df['inst_ownership'].round(2).astype(str) + '%'
        df['growth_score'] = df['growth_score'].round(3)
        df['risk_factor'] = df['risk_factor'].round(3)
        df['final_score'] = df['final_score'].round(3)
        
        print("\n=== Top Long-term Growth Candidates ===")
        print(f"Total stocks analyzed: {len(all_tickers)}")
        print(f"Candidates meeting criteria: {len(growth_candidates)}")
        print("\nTop Candidates by Growth Potential:")
        display_columns = ['ticker', 'category', 'market_cap', 'current_price', 
                         'revenue_growth', 'profit_margin', 'roe', 'forward_pe', 
                         '5y_return', 'final_score']
        print(df[display_columns].head(max_stocks).to_string(index=False))
        
        # Category analysis
        print("\nCategory Performance:")
        category_stats = df.groupby('category')['final_score'].agg(['mean', 'count', 'max'])
        print(category_stats.sort_values('mean', ascending=False))
        
    return [stock['ticker'] for stock in growth_candidates[:max_stocks]]