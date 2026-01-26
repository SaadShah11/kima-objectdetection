"""
Configuration file for Kima Object Detection Training Pipeline

This module contains all configuration settings including:
- Dataset configurations
- Model architectures
- Training hyperparameters
- Workflow flags
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Dataset Configurations
# You can define multiple dataset configurations and switch between them.
# The 'ALIAS' key can be used to refer to a specific dataset later.
# Replace 'YOUR_API_KEY_HERE' with your actual Roboflow API key or use environment variables.
DATASET_CONFIGS = [
    {
        'ROBOFLOW_API_KEY': os.getenv('ROBOFLOW_API_KEY'),  # Use environment variable for security
        'WORKSPACE': 'ball-and-hoop-detection',
        'PROJECT': 'basketball-v2-final-tf-lite-knkc3',
        'VERSION': 2,
        'ALIAS': 'basketball_dataset'
    },
    # Add more dataset configurations here if needed
]

# Select the active dataset configuration by its ALIAS
ACTIVE_DATASET_ALIAS = 'basketball_dataset'

# 2. Model Architectures for Comparison
# Define a list of YOLO model variants (e.g., 'yolo11n', 'yolo11s') to train and compare.
# Ensure these models are compatible with Ultralytics.
MODEL_ARCHITECTURES = ['yolo11n', 'yolo11s']
# MODEL_ARCHITECTURES = ['yolo11n', 'yolo11s', 'yolov8n', 'yolov10n']

# 3. Training Hyperparameters
# These parameters will be applied to all models during training.
TRAINING_HYPERPARAMETERS = {
    'epochs': 1,    # Number of training epochs. Use a smaller number (e.g., 1-5) for quick tests.
    # 'epochs': 60,    # Number of training epochs. Use a smaller number (e.g., 1-5) for quick tests.
    'imgsz': 416,    # Image size for training and inference. 416 for mobile, 640 for higher accuracy.
    'batch': 16,     # Batch size. Adjust based on GPU memory.
    'patience': 10,  # Early stopping patience. Number of epochs to wait for improvement.
    'device': 'auto',  # Device selection: 'auto' (detect and use GPU if available, else CPU), 'cpu', or GPU index (0, 1, 2, etc.)
}

# 4. Workflow Flags
# Control which parts of the workflow to execute.
GENERATE_PLOTS = True                  # Generate and display dataset analysis plots
SAVE_TEST_PREDICTION_IMAGES = True     # Save predicted images during visual testing
PERFORM_MODEL_COMPARISON = True        # Run multi-model comparison training and evaluation

# 5. Paths Configuration
# Base directories (will be set in paths.py, but can be overridden here)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUTS_DIR = DATA_DIR / 'outputs'
DATASETS_DIR = DATA_DIR / 'datasets'
RUNS_DIR = DATA_DIR / 'runs' / 'all_trainings'
