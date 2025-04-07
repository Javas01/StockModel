import os
import pickle
import torch
from classes.StockLSTM import StockLSTM

# Set device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load(ticker):
    # Create full paths
    model_dir = os.path.join(os.getcwd(), 'saved_models', 'pennystockmodel')
    print(f"model_dir {model_dir}")
    model_path = os.path.join(model_dir, f'model.pth')
    scaler_path = os.path.join(model_dir, f'scaler.pkl')
    data_path = os.path.join(model_dir, f'last_data.pkl')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
        print(f"No saved model found for {ticker}")
        return None, None, None
    
    try:
        # Load the saved model data
        checkpoint = torch.load(model_path)
        
        # Create model with saved parameters
        model = StockLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=1
        ).to(device)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        # Load scaler and last data
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(data_path, 'rb') as f:
            last_data = pickle.load(f)
        
        print(f"Model loaded for {ticker}")
        return model, scaler, last_data
        
    except Exception as e:
        print(f"Error loading model for {ticker}: {str(e)}")
        return None, None, None