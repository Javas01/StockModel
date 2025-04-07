import os
import pickle

import torch

def save(model, model_name, scaler, enhanced_data):
    # Create full path for models directory
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
    
    # Create all necessary directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Create full paths for saving
    model_path = os.path.join(model_dir, f'model.pth')
    scaler_path = os.path.join(model_dir, f'scaler.pkl')
    data_path = os.path.join(model_dir, f'last_data.pkl')
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.lstm1.input_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers
    }, model_path)
    
    # Save scaler and data
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(data_path, 'wb') as f:
        pickle.dump(enhanced_data.tail(20), f)
    
    print(f"Model and data saved: {model_name} in {model_dir}")