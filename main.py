"""
Main entry point for Kima Object Detection Training Pipeline

This script orchestrates the entire workflow:
1. Dataset preparation and analysis
2. Model training, validation, and export
3. Comparison reporting and visualization
4. Output archiving
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from src import data_processing
from src import model_training
from src import visualization
from src import archiving
import torch


def main():
    """
    Main function to run the complete training pipeline.
    """
    print("="*80)
    print("KIMA OBJECT DETECTION TRAINING PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Download and prepare datasets from Roboflow")
    print("  2. Analyze dataset characteristics")
    print("  3. Train YOLO models on each dataset")
    print("  4. Export models to CoreML and TFLite formats")
    print("  5. Generate comparison reports and visualizations")
    print("  6. Archive all outputs")
    print("="*80)
    
    # Check device availability early
    device_preference = config.TRAINING_HYPERPARAMETERS.get('device', 'auto')
    device_string, device_info = model_training.detect_and_select_device(device_preference)
    print("\n" + "="*80)
    print("SYSTEM DEVICE DETECTION")
    print("="*80)
    model_training.print_device_info(device_info)
    print(f"Device preference from config: {device_preference}")
    print(f"Selected device for training: {device_info['device_type']}")
    if device_info['device_type'] == 'GPU' and 'device_name' in device_info:
        print(f"GPU Name: {device_info['device_name']}")
    print("="*80 + "\n")
    
    # Step 1: Prepare all datasets
    print("\n[STEP 1] Preparing datasets...")
    prepared_datasets_info = data_processing.prepare_all_datasets()
    
    if not prepared_datasets_info:
        print("ERROR: No datasets were successfully prepared. Exiting.")
        return
    
    # Step 2: Train all models (if enabled)
    if config.PERFORM_MODEL_COMPARISON:
        print("\n[STEP 2] Training models...")
        all_results = model_training.train_all_models(prepared_datasets_info)
        
        # Step 3: Generate comparison report
        print("\n[STEP 3] Generating comparison report...")
        df_comparison = visualization.generate_comparison_report(all_results)
        
        # Step 4: Archive outputs
        print("\n[STEP 4] Archiving outputs...")
        archiving.archive_outputs(all_results)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nAll outputs have been saved to: {config.DATA_DIR / 'outputs'}")
        print(f"Comparison results: {config.DATA_DIR / 'outputs' / 'model_comparison_results.csv'}")
        print(f"Exported models: {config.DATA_DIR / 'outputs' / 'models'}")
        print("="*80)
    else:
        print("\nModel comparison is disabled. Skipping training step.")
        print("Set PERFORM_MODEL_COMPARISON = True in config.py to enable training.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
