"""
Model training module for Kima Object Detection Training Pipeline

This module handles model training, validation, and export.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random
import config
import paths
import utils


def train_and_export_model(model_arch, dataset_info, all_results):
    """
    Train, validate, and export a model for a specific dataset.
    
    Args:
        model_arch: Model architecture name (e.g., 'yolo11n')
        dataset_info: Dictionary containing dataset information
        all_results: List to append results to
        
    Returns:
        Dictionary with training results
    """
    dataset_alias = dataset_info['alias']
    data_yaml_path = dataset_info['data_yaml_path']
    class_names = dataset_info['class_names']
    test_images_path = dataset_info['test_images_path']
    val_images_path = dataset_info['val_images_path']
    
    print(f"\n{'-'*60}")
    print(f"Training Model: {model_arch} on Dataset: {dataset_alias}")
    print(f"{'-'*60}\n")
    
    # Create a unique run name for this model-dataset combination
    run_name = f"{model_arch}_{dataset_alias}"
    model_output_dir = paths.path_manager.get_model_output_dir(model_arch, dataset_alias)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    current_run_info = {
        'dataset_alias': dataset_alias,
        'model_architecture': model_arch,
        'status': 'Failed',
        'mAP50': 0.0,
        'mAP50-95': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'Training Time (min)': 0.0,
        'Inference Time (ms)': 0.0,
        'Trained Model Path': 'N/A',
        'CoreML Export Path': 'N/A',
        'TFLite Export Path': 'N/A',
        'Visual Test Images Path': 'N/A'
    }
    
    try:
        # 1. Load the pre-trained YOLO model
        print(f"Loading model: {model_arch}.pt")
        model = YOLO(f"{model_arch}.pt")
        print(f"Model {model_arch}.pt loaded successfully.")
        
        # 2. Configure training parameters
        training_params = config.TRAINING_HYPERPARAMETERS.copy()
        training_params['data'] = str(data_yaml_path)
        training_params['project'] = str(paths.path_manager.runs_dir.parent)
        training_params['name'] = run_name
        
        print("Starting model training...")
        start_time = time.time()
        # 3. Train the model
        results = model.train(**training_params)
        training_time = (time.time() - start_time) / 60.0  # minutes
        print(f"Training completed for {model_arch} on {dataset_alias} in {training_time:.2f} minutes.")
        
        # Update trained model path
        best_model_pt_path = Path(results.save_dir) / "weights" / "best.pt"
        current_run_info['Trained Model Path'] = str(best_model_pt_path)
        current_run_info['Training Time (min)'] = round(training_time, 2)
        
        # 4. Validate the trained model
        print("Validating model...")
        val_results = model.val()
        current_run_info['mAP50'] = round(val_results.box.map50, 4)
        current_run_info['mAP50-95'] = round(val_results.box.map, 4)
        current_run_info['Precision'] = round(val_results.box.mp, 4)
        current_run_info['Recall'] = round(val_results.box.mr, 4)
        current_run_info['Inference Time (ms)'] = round(val_results.speed['inference'], 2)
        print(f"Validation complete. mAP50: {current_run_info['mAP50']:.4f}")
        
        # 5. Export the best trained model to CoreML and TFLite
        print("Exporting model to CoreML...")
        coreml_export_path = model.export(
            format="coreml",
            imgsz=config.TRAINING_HYPERPARAMETERS['imgsz'],
            half=True,
            nms=True,
            optimize=True
        )
        current_run_info['CoreML Export Path'] = str(coreml_export_path)
        print(f"CoreML model exported to: {coreml_export_path}")
        
        print("Exporting model to TFLite...")
        tflite_export_path = model.export(
            format="tflite",
            imgsz=config.TRAINING_HYPERPARAMETERS['imgsz'],
            int8=True,
            nms=True
        )
        current_run_info['TFLite Export Path'] = str(tflite_export_path)
        print(f"TFLite model exported to: {tflite_export_path}")
        
        # Verify TFLite model metadata
        print("Verifying TFLite model metadata...")
        try:
            utils.verify_and_add_tflite_metadata(
                tflite_path=tflite_export_path,
                class_names=class_names,
                input_size=config.TRAINING_HYPERPARAMETERS['imgsz']
            )
            print("✓ TFLite model metadata verified/added successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not verify/add metadata: {e}")
            print("  Model may still work, but may not be compatible with Task Vision API")
        
        # Store model file paths for later copying
        coreml_path = Path(coreml_export_path) if isinstance(coreml_export_path, str) else coreml_export_path
        tflite_path = Path(tflite_export_path) if isinstance(tflite_export_path, str) else tflite_export_path
        
        current_run_info['_coreml_file_path'] = coreml_path if coreml_path.exists() else None
        current_run_info['_tflite_file_path'] = tflite_path if tflite_path.exists() else None
        current_run_info['_model_name'] = run_name
        current_run_info['_dataset_alias'] = dataset_alias
        
        # 6. Generate and save visual test results
        if config.SAVE_TEST_PREDICTION_IMAGES:
            visual_output_dir = paths.path_manager.get_visual_tests_dir(model_arch, dataset_alias)
            visual_output_dir.mkdir(parents=True, exist_ok=True)
            current_run_info['Visual Test Images Path'] = str(visual_output_dir)
            
            print("Generating visual test results...")
            source_images_dir = test_images_path if test_images_path.exists() else val_images_path
            
            if source_images_dir.exists():
                test_image_files = list(source_images_dir.glob('*.jpg')) + list(source_images_dir.glob('*.png'))
                num_samples = min(5, len(test_image_files))
                if num_samples > 0:
                    sample_images = random.sample(test_image_files, num_samples)
                    
                    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
                    if num_samples == 1:  # Handle single subplot case
                        axes = [axes]
                    fig.suptitle(f'{model_arch} on {dataset_alias} - Sample Predictions', fontsize=16)
                    
                    for i, img_path in enumerate(sample_images):
                        pred_results = model.predict(
                            str(img_path), 
                            conf=0.25, 
                            imgsz=config.TRAINING_HYPERPARAMETERS['imgsz'], 
                            verbose=False
                        )
                        annotated_frame = pred_results[0].plot()  # YOLO's plot method
                        axes[i].imshow(annotated_frame)
                        axes[i].set_title(img_path.name, fontsize=10)
                        axes[i].axis('off')
                        
                        # Log detections for each image
                        detections_count = len(pred_results[0].boxes)
                        print(f"  - {img_path.name}: {detections_count} detections")
                    
                    plt.tight_layout()
                    plt.savefig(visual_output_dir / 'sample_predictions.png')
                    plt.close(fig)
                    print(f"Sample predictions saved to: {visual_output_dir / 'sample_predictions.png'}")
                else:
                    print(f"No suitable images found in {source_images_dir} for visual testing.")
            else:
                print(f"Warning: Test/Validation images directory not found: {source_images_dir}. Skipping visual tests.")
        
        current_run_info['status'] = 'Success'
        
    except Exception as e:
        print(f"Error during processing {model_arch} on {dataset_alias}: {e}")
        import traceback
        traceback.print_exc()
        current_run_info['error_message'] = str(e)
        current_run_info['status'] = 'Failed'
    
    finally:
        all_results.append(current_run_info)
        print(f"Finished processing {model_arch} on {dataset_alias}. Status: {current_run_info['status']}")
    
    return current_run_info


def train_all_models(prepared_datasets_info):
    """
    Train all models on all prepared datasets.
    
    Args:
        prepared_datasets_info: List of processed dataset information dictionaries
        
    Returns:
        List of training results for all model-dataset combinations
    """
    all_results_for_comparison = []
    
    print(f"\n{'='*80}")
    print("STARTING MODEL TRAINING, VALIDATION, AND EXPORT LOOP")
    print(f"{'='*80}\n")
    
    # Outer loop: Iterate through each prepared dataset
    for dataset_info in prepared_datasets_info:
        dataset_alias = dataset_info['alias']
        data_yaml_path = dataset_info['data_yaml_path']
        
        print(f"\n{'*'*70}")
        print(f"Processing Dataset: {dataset_alias} ({data_yaml_path})")
        print(f"{'*'*70}\n")
        
        # Inner loop: Iterate through each model architecture
        for model_arch in config.MODEL_ARCHITECTURES:
            train_and_export_model(model_arch, dataset_info, all_results_for_comparison)
    
    print(f"\n{'='*80}")
    print("COMPLETED ALL MODEL TRAINING, VALIDATION, AND EXPORT LOOPS")
    print(f"{'='*80}\n")
    
    return all_results_for_comparison
