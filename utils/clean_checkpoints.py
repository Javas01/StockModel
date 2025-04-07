import json
import os
import shutil

def clean_checkpoints():
    checkpoint_dir = '.ipynb_checkpoints'
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Removed {checkpoint_dir}")