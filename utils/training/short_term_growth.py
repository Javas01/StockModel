import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.models.save import save
from utils.models.train import train
from classes.StockLSTM import StockLSTM
from utils.stocks.download_stock_data import download_stock_data
from utils.stocks.add_options_indicators import add_option_indicators
from utils.stocks.create_sliding_windows import create_sliding_windows

# Set device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def short_term_growth(tickers, model_name="short_term_model"):
    """
    Train a model using data from multiple tickers to predict short-term stock price movements
    over 1-week to 1-month breakout indicators.
    
    Args:
        tickers (list): List of ticker symbols to use for training
        model_name (str): Name to save the model under
    """
    print(f"Training short-term model using {len(tickers)} tickers")
    
    # Collect data from all tickers
    all_features = []
    all_targets = []
    
    # Define available features for predicting short-term breakout
    available_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',  # Basic price and volume data
        '1w_return', '2w_return', '1m_return',  # Short-term returns
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',  # Short-term momentum indicators
        'BBL', 'BBM', 'BBU',  # Bollinger Bands (breakout indicators)
        '20d_vol',  # Short-term volatility (20-day)
        'above_50ma',  # 50-day moving average (for trend confirmation)
    ]
    
    try:
        # Set date range (use recent data, e.g., 1-2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data
        
        # Collect data from all tickers
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            
            # Download and prepare data
            stock_data = download_stock_data(ticker, start_date, end_date)
            if stock_data.empty:
                print(f"No data downloaded for {ticker}, skipping")
                continue
                
            # Add indicators and handle missing values
            enhanced_data = add_option_indicators(stock_data, ticker)
            enhanced_data = enhanced_data.ffill().bfill()
            enhanced_data = enhanced_data.dropna()
            
            if len(enhanced_data) == 0:
                print(f"No valid data after preprocessing for {ticker}, skipping")
                continue
            
            # Select features and scale
            features = enhanced_data[available_features].values
            
            # Create sliding windows (small window size for short-term prediction)
            window_size = 20  # Small window size to capture short-term trends
            X, y = create_sliding_windows(features, window_size)
            
            if len(X) > 0 and len(y) > 0:
                all_features.append(X)
                all_targets.append(y)
                print(f"Added {len(X)} sequences from {ticker}")
        
        if not all_features:
            raise ValueError("No valid data collected from any ticker")
        
        # Combine all data
        X = np.concatenate(all_features)
        y = np.concatenate(all_targets)
        
        # Define the target as the stock price change over the next 1-week to 1-month (target price)
        y_short_term_change = []
        for i in range(len(y)):
            # Predict the price change in the next 1-4 weeks (adjust target accordingly)
            future_price = y[i, -1]  # Assume the future price is at the last data point in the sequence
            price_1_week_later = enhanced_data['Close'].iloc[i + 5]  # Price ~1 week later
            price_1_month_later = enhanced_data['Close'].iloc[i + 20]  # Price ~1 month later
            
            # Calculate the price change (we could use either 1-week or 1-month for target)
            # For example, predict 1-week change
            change = (price_1_week_later - future_price) / future_price  # Price change rate
            y_short_term_change.append(change)
        
        # Reshape y to reflect short-term price change
        y = np.array(y_short_term_change).reshape(-1, 1)
        
        # Scale all features together
        scaler = MinMaxScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Add validation set
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        # Model parameters
        input_size = len(available_features)
        hidden_size = 128
        num_layers = 2
        output_size = 1  # Predicting short-term price change as a single value
        
        # Initialize model
        model = StockLSTM(input_size, hidden_size, num_layers, output_size).to(device)
        
        # Training parameters
        learning_rate = 0.0001
        num_epochs = 50
        batch_size = 32
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), 
                                    torch.FloatTensor(y_train).to(device))
        val_dataset = TensorDataset(torch.FloatTensor(X_val).to(device), 
                                  torch.FloatTensor(y_val).to(device))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # Train the model
        print("\nStarting model training...")
        train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
        
        # Save the model
        save(model, model_name, scaler, enhanced_data)
        
        print(f"\nTraining completed and model saved as {model_name}")
        return model, scaler
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return None, None
