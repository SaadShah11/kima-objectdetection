"""
Utility functions for Kima Object Detection Training Pipeline

This module contains helper functions used throughout the pipeline.
"""

import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict
import os


def get_class_counts(label_dir, class_names):
    """
    Analyze YOLO format label files to count instances of each class.
    
    Args:
        label_dir: Path to directory containing label files
        class_names: List of class names
        
    Returns:
        tuple: (class_counts dict, total_annotations int)
    """
    class_counts = defaultdict(int)
    total_annotations = 0
    if not label_dir.exists():
        print(f"Warning: Label directory not found: {label_dir}")
        return class_counts, total_annotations

    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
                            total_annotations += 1
                    except ValueError:
                        print(f"Warning: Invalid class_id in file {label_file.name}, line: {line.strip()}")
    return class_counts, total_annotations


def plot_image_with_bboxes(image_path, label_path, class_names, save_path):
    """
    Visualize an image with its corresponding YOLO bounding box annotations.
    
    Args:
        image_path: Path to image file
        label_path: Path to label file
        class_names: List of class names
        save_path: Path to save the plotted image
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    img_width, img_height = img.size

    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        # YOLO polygon format: class_id x1 y1 x2 y2 ...
                        points_normalized = [float(p) for p in parts[1:]]

                        if len(points_normalized) % 2 == 0 and len(points_normalized) > 0:
                            x_coords_norm = points_normalized[0::2]
                            y_coords_norm = points_normalized[1::2]

                            min_x_norm = min(x_coords_norm)
                            max_x_norm = max(x_coords_norm)
                            min_y_norm = min(y_coords_norm)
                            max_y_norm = max(y_coords_norm)

                            x_min_px = min_x_norm * img_width
                            y_min_px = min_y_norm * img_height
                            width_px = (max_x_norm - min_x_norm) * img_width
                            height_px = (max_y_norm - min_y_norm) * img_height

                            rect = patches.Rectangle((x_min_px, y_min_px), width_px, height_px,
                                                   linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)

                            if 0 <= class_id < len(class_names):
                                label = class_names[class_id]
                            else:
                                label = f'Class {class_id}'
                            ax.text(x_min_px, y_min_px - 5, label, color='r', fontsize=10, 
                                  bbox=dict(facecolor='white', alpha=0.7))
                        else:
                            print(f"Warning: Invalid number of coordinates in label file {label_path.name} for line: {line.strip()}")
                    except ValueError:
                        print(f"Warning: Invalid data in label file {label_path.name}, line: {line.strip()}")
    else:
        print(f"Warning: Label file not found for {image_path.name}")

    ax.set_title(image_path.name)
    ax.axis('off')
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory


def get_dataset_stats(image_dir, label_dir):
    """
    Get basic statistics on the number of images and label files.
    
    Args:
        image_dir: Path to directory containing images
        label_dir: Path to directory containing label files
        
    Returns:
        tuple: (num_images int, num_labels int)
    """
    num_images = len(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    num_labels = len(list(label_dir.glob('*.txt')))
    return num_images, num_labels


def verify_and_add_tflite_metadata(tflite_path, class_names, input_size):
    """
    Verify TFLite model metadata and provide compatibility information.
    
    Args:
        tflite_path: Path to the TFLite model file
        class_names: List of class names for the model
        input_size: Input image size (e.g., 416)
        
    Returns:
        bool or None: True if metadata found, False if not found, None if verification failed
    """
    try:
        # Try to import TensorFlow Lite
        try:
            import tensorflow as tf
        except ImportError:
            print("  ⚠ TensorFlow not available. Installing...")
            os.system("pip install tensorflow")
            import tensorflow as tf

        # Load and inspect the TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\n  Model Structure Analysis:")
        print(f"  - Input tensors: {len(input_details)}")
        for i, inp in enumerate(input_details):
            print(f"    Input {i}: shape={inp['shape']}, dtype={inp['dtype']}, name={inp.get('name', 'N/A')}")

        print(f"  - Output tensors: {len(output_details)}")
        for i, out in enumerate(output_details):
            print(f"    Output {i}: shape={out['shape']}, dtype={out['dtype']}, name={out.get('name', 'N/A')}")

        # Check if metadata exists using tflite_support (if available)
        try:
            from tflite_support import metadata
            displayer = metadata.MetadataDisplayer.with_model_file(str(tflite_path))
            metadata_json = displayer.get_metadata_json()

            if metadata_json:
                print(f"  ✓ Metadata found in model")
                # Parse metadata to check for ObjectDetector compatibility
                import json
                meta = json.loads(metadata_json)
                if 'subgraph_metadata' in meta and len(meta['subgraph_metadata']) > 0:
                    print(f"  ✓ Model has subgraph metadata")
                    # Check for input/output metadata
                    subgraph = meta['subgraph_metadata'][0]
                    if 'input_tensor_metadata' in subgraph:
                        print(f"  ✓ Input tensor metadata present")
                    if 'output_tensor_metadata' in subgraph:
                        print(f"  ✓ Output tensor metadata present")
                return True
            else:
                print(f"  ⚠ No metadata found in model")
                print(f"  ⚠ Model may not be compatible with TensorFlow Lite Task Vision ObjectDetector API")
                print(f"  ℹ Consider using TensorFlow Lite Model Maker or adding metadata manually")
                return False
        except ImportError:
            print(f"  ⚠ tflite_support library not available for metadata verification")
            print(f"  ℹ Install with: pip install tflite-support")
            print(f"  ℹ Model structure looks correct, but metadata compatibility cannot be verified")
            print(f"  ℹ Note: YOLO models exported with Ultralytics may not include metadata")
            print(f"  ℹ If ObjectDetector API fails, consider using Interpreter API directly")
            return None
        except Exception as e:
            print(f"  ⚠ Could not read metadata: {e}")
            print(f"  ℹ Model may work, but metadata verification failed")
            print(f"  ℹ If ObjectDetector API fails, the model can still be used with Interpreter API")
            return None

    except Exception as e:
        print(f"  ✗ Error verifying model: {e}")
        print(f"  ℹ Model file exists but could not be verified")
        print(f"  ℹ Model dimensions: {input_size}x{input_size}, Classes: {len(class_names)}")
        return False
