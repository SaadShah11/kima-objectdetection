"""
Archiving module for Kima Object Detection Training Pipeline

This module handles output archiving and model file organization.
"""

import shutil
from pathlib import Path
import paths


def archive_outputs(all_results):
    """
    Consolidate all artifacts and create archive.
    
    Args:
        all_results: List of training results dictionaries
    """
    outputs_base_dir = paths.path_manager.outputs_dir
    outputs_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Consolidating all artifacts into '{outputs_base_dir}'...")
    
    # Move the training runs directory into outputs
    try:
        trainings_run_source = paths.path_manager.runs_dir
        if trainings_run_source.exists():
            destination_path = outputs_base_dir / 'all_trainings'
            if destination_path.exists():
                shutil.rmtree(destination_path)
            shutil.copytree(trainings_run_source, destination_path)
            print(f"Moved '{trainings_run_source}' to '{destination_path}'")
        else:
            print(f"Warning: '{trainings_run_source}' not found. Skipping move.")
    except Exception as e:
        print(f"Error moving training runs: {e}")
    
    # Create models directory and copy exported model files
    models_output_dir = paths.path_manager.get_models_dir()
    models_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying exported model files to '{models_output_dir}'...")
    
    models_copied = 0
    for result in all_results:
        if result.get('status') == 'Success':
            model_name = result.get('_model_name', result.get('model_architecture', 'unknown'))
            dataset_alias = result.get('_dataset_alias', result.get('dataset_alias', 'unknown'))
            
            # Copy TFLite model
            tflite_source = result.get('_tflite_file_path')
            if tflite_source:
                if isinstance(tflite_source, str):
                    tflite_source = Path(tflite_source)
                if tflite_source.exists():
                    tflite_dest = models_output_dir / f"{model_name}_{dataset_alias}.tflite"
                    try:
                        shutil.copy2(tflite_source, tflite_dest)
                        print(f"  ✓ Copied TFLite: {tflite_dest.name} ({tflite_dest.stat().st_size / (1024*1024):.2f} MB)")
                        models_copied += 1
                    except Exception as e:
                        print(f"  ✗ Failed to copy TFLite model {tflite_source}: {e}")
                else:
                    print(f"  ⚠ TFLite model not found at: {tflite_source}")
            else:
                print(f"  ⚠ No TFLite path stored for {model_name} on {dataset_alias}")
            
            # Copy CoreML model (mlpackage is a directory, so use copytree)
            coreml_source = result.get('_coreml_file_path')
            if coreml_source:
                if isinstance(coreml_source, str):
                    coreml_source = Path(coreml_source)
                if coreml_source.exists():
                    coreml_dest = models_output_dir / f"{model_name}_{dataset_alias}.mlpackage"
                    try:
                        if coreml_dest.exists():
                            shutil.rmtree(coreml_dest)
                        if coreml_source.is_dir():
                            shutil.copytree(coreml_source, coreml_dest)
                            # Calculate size for directory
                            total_size = sum(f.stat().st_size for f in coreml_dest.rglob('*') if f.is_file())
                            print(f"  ✓ Copied CoreML: {coreml_dest.name} ({total_size / (1024*1024):.2f} MB)")
                        else:
                            shutil.copy2(coreml_source, coreml_dest)
                            print(f"  ✓ Copied CoreML: {coreml_dest.name} ({coreml_dest.stat().st_size / (1024*1024):.2f} MB)")
                        models_copied += 1
                    except Exception as e:
                        print(f"  ✗ Failed to copy CoreML model {coreml_source}: {e}")
                else:
                    print(f"  ⚠ CoreML model not found at: {coreml_source}")
            else:
                print(f"  ⚠ No CoreML path stored for {model_name} on {dataset_alias}")
    
    if models_copied > 0:
        print(f"\n✓ Successfully copied {models_copied} model file(s) to '{models_output_dir}'")
        print(f"  Models are ready for use in your Android/iOS project:")
        print(f"    - TFLite files (.tflite) → Android assets folder")
        print(f"    - CoreML files (.mlpackage) → iOS Xcode project")
    else:
        print(f"\n⚠ No model files were copied. Check if models were exported successfully.")
        print(f"  Check the 'all_trainings' directory for exported models.")
    
    # Create a zip archive of the entire 'outputs' directory
    zip_file_name = paths.path_manager.get_zip_archive_path()
    print(f"\nCreating zip archive '{zip_file_name}'...")
    try:
        shutil.make_archive(str(zip_file_name).replace('.zip', ''), 'zip', outputs_base_dir)
        print(f"Archive '{zip_file_name}' created successfully.")
    except Exception as e:
        print(f"Error creating zip archive: {e}")
