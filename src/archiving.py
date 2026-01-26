"""
Archiving module for Kima Object Detection Training Pipeline

This module handles output archiving and model file organization.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent))

import shutil
import paths
import config


def archive_outputs(all_results):
    """
    Consolidate all artifacts and create archive.
    
    Args:
        all_results: List of training results dictionaries
    """
    outputs_base_dir = paths.path_manager.outputs_dir
    outputs_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Consolidating all artifacts into '{outputs_base_dir}'...")
    
    # Move the training runs directory into outputs (always overwrite)
    try:
        trainings_run_source = paths.path_manager.runs_dir
        if trainings_run_source.exists():
            destination_path = outputs_base_dir / 'all_trainings'
            # Always overwrite existing training runs
            if destination_path.exists():
                print(f"Removing existing training runs directory: {destination_path}")
                shutil.rmtree(destination_path)
            shutil.copytree(trainings_run_source, destination_path)
            print(f"✓ Copied training runs from '{trainings_run_source}' to '{destination_path}'")
        else:
            print(f"Warning: '{trainings_run_source}' not found. Skipping copy.")
    except Exception as e:
        print(f"Error copying training runs: {e}")
    
    # Create models directory and copy exported model files
    models_output_dir = paths.path_manager.get_models_dir()
    models_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying exported model files to '{models_output_dir}'...")
    
    models_copied = 0
    for result in all_results:
        if result.get('status') == 'Success':
            model_name = result.get('_model_name', result.get('model_architecture', 'unknown'))
            dataset_alias = result.get('_dataset_alias', result.get('dataset_alias', 'unknown'))
            
            # Models should already be in organized directories from model_training.py
            # This section is kept for backward compatibility but models are already organized
            android_dir = paths.path_manager.get_android_models_dir()
            ios_dir = paths.path_manager.get_ios_models_dir()
            android_dir.mkdir(parents=True, exist_ok=True)
            ios_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy TFLite model to Android directory if not already there
            tflite_source = result.get('_tflite_file_path')
            if tflite_source:
                if isinstance(tflite_source, str):
                    tflite_source = Path(tflite_source)
                if tflite_source.exists():
                    # Always copy/overwrite to ensure latest version
                    # Get image size from config for filename
                    imgsz = config.TRAINING_HYPERPARAMETERS.get('imgsz', 416)
                    tflite_dest = android_dir / f"{model_name}_imgsz{imgsz}.tflite"
                    try:
                        # Remove existing file if it exists
                        if tflite_dest.exists():
                            tflite_dest.unlink()
                        # Copy to destination (overwrites if exists)
                        shutil.copy2(tflite_source, tflite_dest)
                        print(f"  ✓ Copied/overwritten TFLite: {tflite_dest.name} ({tflite_dest.stat().st_size / (1024*1024):.2f} MB)")
                        models_copied += 1
                    except Exception as e:
                        print(f"  ✗ Failed to copy TFLite model {tflite_source}: {e}")
                else:
                    print(f"  ⚠ TFLite model not found at: {tflite_source}")
            else:
                print(f"  ⚠ No TFLite path stored for {model_name} on {dataset_alias}")
            
            # Copy CoreML model to iOS directory if not already there
            coreml_source = result.get('_coreml_file_path')
            if coreml_source:
                if isinstance(coreml_source, str):
                    coreml_source = Path(coreml_source)
                if coreml_source.exists():
                    # Always copy/overwrite to ensure latest version
                    # Get image size from config for filename
                    imgsz = config.TRAINING_HYPERPARAMETERS.get('imgsz', 416)
                    coreml_dest = ios_dir / f"{model_name}_imgsz{imgsz}.mlpackage"
                    try:
                        # Remove existing file/directory if it exists
                        if coreml_dest.exists():
                            if coreml_dest.is_dir():
                                shutil.rmtree(coreml_dest)
                            else:
                                coreml_dest.unlink()
                        # Copy to destination (overwrites if exists)
                        if coreml_source.is_dir():
                            shutil.copytree(coreml_source, coreml_dest)
                            # Calculate size for directory
                            total_size = sum(f.stat().st_size for f in coreml_dest.rglob('*') if f.is_file())
                            print(f"  ✓ Copied/overwritten CoreML: {coreml_dest.name} ({total_size / (1024*1024):.2f} MB)")
                        else:
                            shutil.copy2(coreml_source, coreml_dest)
                            print(f"  ✓ Copied/overwritten CoreML: {coreml_dest.name} ({coreml_dest.stat().st_size / (1024*1024):.2f} MB)")
                        models_copied += 1
                    except Exception as e:
                        print(f"  ✗ Failed to copy CoreML model {coreml_source}: {e}")
                else:
                    print(f"  ⚠ CoreML model not found at: {coreml_source}")
            else:
                print(f"  ⚠ No CoreML path stored for {model_name} on {dataset_alias}")
    
    if models_copied > 0:
        android_dir = paths.path_manager.get_android_models_dir()
        ios_dir = paths.path_manager.get_ios_models_dir()
        print(f"\n✓ Successfully copied {models_copied} model file(s)")
        print(f"  Models are organized and ready for use:")
        print(f"    - Android models (TFLite): {android_dir}")
        print(f"    - iOS models (CoreML): {ios_dir}")
        print(f"    - TFLite files (.tflite) → Copy to Android assets folder")
        print(f"    - CoreML files (.mlpackage) → Add to iOS Xcode project")
    else:
        print(f"\n⚠ No model files were copied. Check if models were exported successfully.")
        print(f"  Check the 'all_trainings' directory for exported models.")
