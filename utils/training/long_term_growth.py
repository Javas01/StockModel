import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import constants
from utils.models.save import save
from utils.models.train import train
from classes.StockLSTM import StockLSTM
from utils.stocks.download_stock_data import download_stock_data
from utils.stocks.add_long_term_stock_features import add_long_term_stock_features
from utils.stocks.create_sliding_windows import create_sliding_windows

# Set device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def long_term_growth(tickers, model_name="growth_model"):
    print(f"Training growth model using {len(tickers)} tickers")
    
    # Collect data from all tickers
    all_features = []
    all_targets = []
    
    try:
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*3)  # 3 years of data
        
        # Collect data from all tickers
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            
            # Download and prepare data
            stock_data = download_stock_data(ticker, start_date, end_date)
            if stock_data.empty:
                print(f"No data downloaded for {ticker}, skipping")
                continue
                
            # Add indicators and handle missing values
            enhanced_data = add_long_term_stock_features(stock_data, ticker)
            if enhanced_data is None:
                print(f"Could not calculate features for {ticker}, skipping")
                continue
                
            enhanced_data = enhanced_data.ffill().bfill()
            enhanced_data = enhanced_data.dropna()
            
            if len(enhanced_data) == 0:
                print(f"No valid data after preprocessing for {ticker}, skipping")
                continue
            
            # Define available features
            long_term_growth_features = constants.LONG_TERM_STOCK_FEATURES
            
            # Select features
            features = enhanced_data[long_term_growth_features].values
            
            # Create sliding windows
            window_size = 90
            X, y = create_sliding_windows(features, window_size)
            
            if len(X) > 0:
                all_features.append(X)
                all_targets.append(y)
                print(f"Added {len(X)} sequences from {ticker}")
        
        if not all_features:
            raise ValueError("No valid data collected from any ticker")
        
        # Combine all data
        X = np.concatenate(all_features)
        y = np.concatenate(all_targets)
        
        # Scale features
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
        input_size = len(long_term_growth_features)
        hidden_size = 128
        num_layers = 2
        output_size = 1
        
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
        
        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        print("\nStarting model training...")
        train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
        
        # Save the model
        save(model, model_name, scaler, enhanced_data)
        
        print(f"\nTraining completed and model saved as {model_name}")
        return model, scaler
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return None, None