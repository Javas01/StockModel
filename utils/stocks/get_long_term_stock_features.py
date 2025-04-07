def get_long_term_stock_features():
    return [
        # Price data (these are definitely available)
        'Open', 'High', 'Low', 'Close', 'Volume',
        
        # Returns (keep only what's being calculated)
        '1w_return', '2w_return', '1m_return',
        
        # Volatility (keep only what's being calculated)
        '20d_vol',
        
        # Technical Indicators (keep only what's being calculated)
        'BBL', 'BBM', 'BBU',  # These appear in your DataFrame
        
        # Moving Averages (adjust to match what's available)
        'above_50ma',
        
        # Fundamental Indicators (keep only what's being calculated)
        'P_B_Ratio'  # This appears in your DataFrame
    ]