import os

import torch

def save_model(model, ticker, scaler, enhanced_data):
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model state dict
    model_path = f'models/{ticker}_model.pth'
    scaler_path = f'models/{ticker}_scaler.pkl'
    data_path = f'models/{ticker}_last_data.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    # Save the scaler and last data point using pickle
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(data_path, 'wb') as f:
        pickle.dump(enhanced_data.tail(20), f)  # Save last 20 days for prediction
    
    print(f"Model saved for {ticker}")