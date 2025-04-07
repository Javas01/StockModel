import json
import os
import shutil

def configure_jupyter():
    jupyter_config_dir = os.path.expanduser('~/.jupyter')
    if not os.path.exists(jupyter_config_dir):
        os.makedirs(jupyter_config_dir)
    
    config_path = os.path.join(jupyter_config_dir, 'jupyter_notebook_config.json')
    
    config = {
        "NotebookApp": {
            "clear_output_before_save": True  # Fixed: true -> True
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)