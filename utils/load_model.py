import os
import torch
from classes.StockLSTM import StockLSTM

def load_model(ticker):
    model_path = f'models/{ticker}_model.pth'
    scaler_path = f'models/{ticker}_scaler.pkl'
    data_path = f'models/{ticker}_last_data.pkl'
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
        print(f"No saved model found for {ticker}")
        return None, None, None
    
    # Initialize model architecture
    input_size = 21  # Number of features in your data
    hidden_size = 128
    num_layers = 2
    output_size = 1
    
    # Create model and load state
    model = StockLSTM(input_size, hidden_size, num_layers, output_size).to(torch.device)
    model.load_state_dict(torch.load(model_path))
    
    # Load scaler and last data
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(data_path, 'rb') as f:
        last_data = pickle.load(f)
    
    print(f"Model loaded for {ticker}")
    return model, scaler, last_data