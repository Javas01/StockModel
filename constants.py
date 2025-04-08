LONG_TERM_STOCK_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    '1w_return', '2w_return', '1m_return',
    '20d_vol',
    'BBL', 'BBM', 'BBU',
    'above_50ma',
    'P_B_Ratio' 
]

SHORT_TERM_STOCK_FEATURES = [
    # Basic price data
    'Open', 'High', 'Low', 'Close', 'Volume',
    
    # Short-term returns
    '1w_return', '2w_return', '1m_return',
    
    # Volatility
    '10d_vol', '20d_vol',
    
    # Volume indicators
    'Volume_SMA_20', 'Volume_SMA_50', 'Volume_Change',
    
    # Technical indicators
    'STOCH_RSI_K', 'STOCH_RSI_D',
    'RSI',
    'MACD', 'MACD_signal', 'MACD_hist',
    'BBL', 'BBM', 'BBU',
    
    # Moving averages and trends
    'above_50ma', 'above_20ma',
    'OBV',
    'EMA_10', 'EMA_20'
]