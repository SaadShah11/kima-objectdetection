"""
Data processing module for Kima Object Detection Training Pipeline

This module handles dataset preparation, download, and analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add src directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import matplotlib.pyplot as plt
import random
import shutil
import os
from roboflow import Roboflow
import config
import paths
import utils


def validate_existing_dataset(dataset_location):
    """
    Validate if an existing dataset directory has the required structure and files.
    
    Args:
        dataset_location: Path to the dataset directory
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if not dataset_location.exists():
        return False, f"Dataset directory does not exist: {dataset_location}"
    
    if not dataset_location.is_dir():
        return False, f"Dataset path is not a directory: {dataset_location}"
    
    # Check for data.yaml file
    data_yaml_path = dataset_location / 'data.yaml'
    if not data_yaml_path.exists():
        return False, f"data.yaml file not found in: {dataset_location}"
    
    # Check for required directories
    required_dirs = {
        'train': ['images', 'labels'],
        'valid': ['images', 'labels'],
        'test': ['images', 'labels']
    }
    
    for split_name, subdirs in required_dirs.items():
        split_dir = dataset_location / split_name
        if not split_dir.exists():
            return False, f"Missing '{split_name}' directory in: {dataset_location}"
        
        for subdir in subdirs:
            subdir_path = split_dir / subdir
            if not subdir_path.exists():
                return False, f"Missing '{split_name}/{subdir}' directory in: {dataset_location}"
            
            # Check if directory has at least some files (not empty)
            if not any(subdir_path.iterdir()):
                return False, f"Empty '{split_name}/{subdir}' directory in: {dataset_location}"
    
    # Check if there are at least some images and labels
    train_images = dataset_location / 'train' / 'images'
    train_labels = dataset_location / 'train' / 'labels'
    
    image_files = list(train_images.glob('*.jpg')) + list(train_images.glob('*.png'))
    label_files = list(train_labels.glob('*.txt'))
    
    if len(image_files) == 0:
        return False, f"No training images found in: {train_images}"
    
    if len(label_files) == 0:
        return False, f"No training labels found in: {train_labels}"
    
    return True, None


def find_existing_dataset(workspace, project_name, version_num):
    """
    Try to find an existing dataset directory that matches the project configuration.
    Checks multiple possible locations where Roboflow might have downloaded the dataset.
    
    Args:
        workspace: Roboflow workspace name
        project_name: Roboflow project name
        version_num: Dataset version number
        
    Returns:
        Path to existing dataset directory if found, None otherwise
    """
    # Roboflow typically creates directories like: {project_name}-{version}
    # Try different possible naming patterns
    possible_names = [
        f"{project_name}-{version_num}",
        f"{project_name}",
        f"{workspace}-{project_name}-{version_num}",
    ]
    
    # Locations to search for existing datasets
    search_locations = [
        paths.path_manager.datasets_dir,  # Preferred location in data/datasets/
        Path.cwd(),  # Current working directory (Roboflow default)
        Path.cwd() / project_name,  # Sometimes in a subdirectory
    ]
    
    # Search in each location with each possible name
    for search_location in search_locations:
        if not search_location.exists():
            continue
            
        for name in possible_names:
            potential_path = search_location / name
            if potential_path.exists() and potential_path.is_dir():
                is_valid, error = validate_existing_dataset(potential_path)
                if is_valid:
                    return potential_path
                else:
                    # Don't print error for every failed check, only if it's close
                    pass
    
    # Also check all subdirectories in datasets_dir (for manually organized datasets)
    datasets_dir = paths.path_manager.datasets_dir
    if datasets_dir.exists():
        for item in datasets_dir.iterdir():
            if item.is_dir():
                is_valid, error = validate_existing_dataset(item)
                if is_valid:
                    # Check if this might be the right dataset by checking data.yaml
                    data_yaml = item / 'data.yaml'
                    if data_yaml.exists():
                        try:
                            with open(data_yaml, 'r') as f:
                                data_content = yaml.safe_load(f)
                                # If it has valid structure, it might be our dataset
                                if 'names' in data_content and 'nc' in data_content:
                                    # For now, return it if structure is valid
                                    # (In a more robust implementation, we'd check project metadata)
                                    return item
                        except Exception:
                            pass
    
    return None


def correct_data_yaml(dataset_location, data_yaml_path):
    """
    Correct the data.yaml file to use relative paths for Ultralytics compatibility.
    
    Args:
        dataset_location: Path to the dataset root directory
        data_yaml_path: Path to the original data.yaml file
        
    Returns:
        tuple: (modified_data_yaml_path, class_names, nc)
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data_config_content = yaml.safe_load(f)
    
    class_names = data_config_content['names']
    nc = data_config_content['nc']
    
    # Construct absolute paths for images and labels
    train_images_abs_path = dataset_location / 'train' / 'images'
    train_labels_abs_path = dataset_location / 'train' / 'labels'
    val_images_abs_path = dataset_location / 'valid' / 'images'
    val_labels_abs_path = dataset_location / 'valid' / 'labels'
    test_images_abs_path = dataset_location / 'test' / 'images'
    test_labels_abs_path = dataset_location / 'test' / 'labels'
    
    # Update data_config with paths RELATIVE to the dataset_root_dir
    data_config_content['train'] = str(train_images_abs_path.relative_to(dataset_location))
    data_config_content['val'] = str(val_images_abs_path.relative_to(dataset_location))
    data_config_content['test'] = str(test_images_abs_path.relative_to(dataset_location))
    
    modified_data_yaml_path = dataset_location / 'data_corrected.yaml'
    with open(modified_data_yaml_path, 'w') as f:
        yaml.dump(data_config_content, f, default_flow_style=False)
    
    return modified_data_yaml_path, class_names, nc


def analyze_dataset(dataset_paths, class_names, output_dir):
    """
    Perform dataset analysis including class distribution and sample images.
    
    Args:
        dataset_paths: Dictionary with paths to train/val/test images and labels
        class_names: List of class names
        output_dir: Directory to save analysis outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Class Distribution Analysis ---")
    # Get counts for training set
    train_class_counts, total_train_annotations = utils.get_class_counts(
        dataset_paths['train_labels'], class_names
    )
    print(f"Training Set Class Distribution (Total Annotations: {total_train_annotations}):")
    for cls, count in train_class_counts.items():
        print(f"  {cls}: {count}")
    
    # Get counts for validation set
    val_class_counts, total_val_annotations = utils.get_class_counts(
        dataset_paths['val_labels'], class_names
    )
    print(f"\nValidation Set Class Distribution (Total Annotations: {total_val_annotations}):")
    for cls, count in val_class_counts.items():
        print(f"  {cls}: {count}")
    
    # Plotting the class distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set plot
    if train_class_counts:
        axes[0].bar(train_class_counts.keys(), train_class_counts.values(), color='skyblue')
        axes[0].set_title('Training Set Class Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Number of Instances')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Validation set plot
    if val_class_counts:
        axes[1].bar(val_class_counts.keys(), val_class_counts.values(), color='lightcoral')
        axes[1].set_title('Validation Set Class Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Number of Instances')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close(fig)
    
    print("\n--- Sample Images with Annotations ---")
    # Get a list of all training image files
    train_image_files = list(dataset_paths['train_images'].glob('*.jpg')) + \
                        list(dataset_paths['train_images'].glob('*.png'))
    
    if len(train_image_files) == 0:
        print(f"No training images found in {dataset_paths['train_images']}")
    else:
        num_samples_to_show = min(5, len(train_image_files))
        selected_images = []
        class_image_map = {name: None for name in class_names}
        
        # Try to find an image for each class
        for img_file in train_image_files:
            label_file = dataset_paths['train_labels'] / (img_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    if class_image_map[class_name] is None:
                                        class_image_map[class_name] = img_file
                                        break  # Move to next image once a class is covered
                            except ValueError:
                                pass  # Ignore invalid lines for selection
        
        # Add images that contain unique classes first
        for class_name in class_names:
            if class_image_map[class_name] is not None and class_image_map[class_name] not in selected_images:
                selected_images.append(class_image_map[class_name])
        
        # Fill the rest with random images if needed
        remaining_slots = num_samples_to_show - len(selected_images)
        if remaining_slots > 0:
            remaining_images = [img for img in train_image_files if img not in selected_images]
            if len(remaining_images) > remaining_slots:
                selected_images.extend(random.sample(remaining_images, remaining_slots))
            else:
                selected_images.extend(remaining_images)
        
        if not selected_images:
            print("Could not find any suitable images to display with annotations.")
        else:
            print(f"Displaying and saving {len(selected_images)} sample images, prioritizing distinct classes.")
            for idx, img_file in enumerate(selected_images):
                label_file = dataset_paths['train_labels'] / (img_file.stem + '.txt')
                save_path = output_dir / f"sample_image_{img_file.stem}.png"
                utils.plot_image_with_bboxes(img_file, label_file, class_names, save_path)
                print(f"  Saved sample image with annotations to {save_path}")


def process_dataset(dataset_config):
    """
    Process a single dataset: download, correct data.yaml, and optionally analyze.
    
    Args:
        dataset_config: Dictionary containing dataset configuration
        
    Returns:
        Dictionary with processed dataset information, or None if processing failed
    """
    rf_api_key = dataset_config['ROBOFLOW_API_KEY']
    workspace = dataset_config['WORKSPACE']
    project_name = dataset_config['PROJECT']
    version_num = dataset_config['VERSION']
    alias = dataset_config['ALIAS']
    
    print(f"\n{'='*80}")
    print(f"Processing Dataset: {alias}")
    print(f"{'='*80}")
    
    try:
        # 1. Check if dataset already exists
        print("Checking for existing dataset...")
        existing_dataset_path = find_existing_dataset(workspace, project_name, version_num)
        dataset_was_existing = existing_dataset_path is not None
        
        if existing_dataset_path:
            print(f"✓ Found existing dataset at: {existing_dataset_path}")
            print("  Dataset validation passed. Skipping download.")
            dataset_location = existing_dataset_path
        else:
            # 2. Download dataset from Roboflow if not found
            if rf_api_key == 'YOUR_API_KEY_HERE' or not rf_api_key:
                print("WARNING: Roboflow API Key not set. Skipping dataset download for this config.")
                return None
            
            print("  No valid existing dataset found. Downloading from Roboflow...")
            
            # Ensure datasets directory exists
            datasets_dir = paths.path_manager.datasets_dir
            datasets_dir.mkdir(parents=True, exist_ok=True)
            
            # Set working directory to datasets_dir for download
            original_cwd = os.getcwd()
            try:
                os.chdir(str(datasets_dir))
                
                rf = Roboflow(api_key=rf_api_key)
                project = rf.workspace(workspace).project(project_name)
                version = project.version(version_num)
                dataset = version.download("yolov11")
                
                # Get the downloaded dataset location (absolute path)
                downloaded_location = Path(dataset.location).resolve()
                
                # If dataset was downloaded to a different location, move it to datasets_dir
                try:
                    # Check if path is relative to datasets_dir
                    if not str(downloaded_location).startswith(str(datasets_dir.resolve())):
                        # Move dataset to datasets_dir
                        target_location = datasets_dir / downloaded_location.name
                        if target_location.exists():
                            shutil.rmtree(target_location)
                        shutil.move(str(downloaded_location), str(target_location))
                        dataset_location = target_location
                        print(f"✓ Dataset moved to: {dataset_location}")
                    else:
                        dataset_location = downloaded_location
                        print(f"✓ Dataset '{alias}' downloaded to: {dataset_location}")
                except AttributeError:
                    # Fallback for older Python versions that don't have is_relative_to
                    if not str(downloaded_location).startswith(str(datasets_dir.resolve())):
                        target_location = datasets_dir / downloaded_location.name
                        if target_location.exists():
                            shutil.rmtree(target_location)
                        shutil.move(str(downloaded_location), str(target_location))
                        dataset_location = target_location
                        print(f"✓ Dataset moved to: {dataset_location}")
                    else:
                        dataset_location = downloaded_location
                        print(f"✓ Dataset '{alias}' downloaded to: {dataset_location}")
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
            
            # Validate the downloaded dataset
            is_valid, error = validate_existing_dataset(dataset_location)
            if not is_valid:
                print(f"WARNING: Downloaded dataset validation failed: {error}")
                print("  Continuing anyway, but dataset may be incomplete.")
        
        # 3. Create dataset-specific output directory
        output_base_dir = paths.path_manager.get_dataset_output_dir(alias)
        dataset_analysis_output_dir = paths.path_manager.get_dataset_analysis_dir(alias)
        dataset_analysis_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dataset analysis outputs will be saved to: {dataset_analysis_output_dir}")
        
        # 4. Correct and save the 'data.yaml' file (or use existing corrected one)
        original_data_yaml_path = dataset_location / 'data.yaml'
        modified_data_yaml_path = dataset_location / 'data_corrected.yaml'
        
        # Check if corrected yaml already exists and dataset was existing
        if modified_data_yaml_path.exists() and dataset_was_existing:
            print(f"Using existing corrected data.yaml: {modified_data_yaml_path}")
            with open(modified_data_yaml_path, 'r') as f:
                data_config_content = yaml.safe_load(f)
            class_names = data_config_content['names']
            nc = data_config_content['nc']
        else:
            # Create corrected yaml
            modified_data_yaml_path, class_names, nc = correct_data_yaml(
                dataset_location, original_data_yaml_path
            )
            print(f"Created/modified data.yaml saved to: {modified_data_yaml_path}")
        
        print(f"Number of classes (nc): {nc}")
        print(f"Class Names: {class_names}")
        
        # Store absolute paths for analysis functions
        dataset_paths = {
            'train_images': dataset_location / 'train' / 'images',
            'train_labels': dataset_location / 'train' / 'labels',
            'val_images': dataset_location / 'valid' / 'images',
            'val_labels': dataset_location / 'valid' / 'labels',
            'test_images': dataset_location / 'test' / 'images',
            'test_labels': dataset_location / 'test' / 'labels',
        }
        
        # 5. Perform and display dataset analysis (if enabled)
        if config.GENERATE_PLOTS:
            analyze_dataset(dataset_paths, class_names, dataset_analysis_output_dir)
        
        # 6. Display dataset statistics
        train_num_images, train_num_labels = utils.get_dataset_stats(
            dataset_paths['train_images'], dataset_paths['train_labels']
        )
        print(f"\nTraining Set: {train_num_images} images, {train_num_labels} label files")
        
        val_num_images, val_num_labels = utils.get_dataset_stats(
            dataset_paths['val_images'], dataset_paths['val_labels']
        )
        print(f"Validation Set: {val_num_images} images, {val_num_labels} label files")
        
        test_num_images, test_num_labels = utils.get_dataset_stats(
            dataset_paths['test_images'], dataset_paths['test_labels']
        )
        print(f"Test Set: {test_num_images} images, {test_num_labels} label files")
        
        # 7. Store the processed dataset information
        dataset_info = {
            'alias': alias,
            'dataset_root_dir': dataset_location,
            'data_yaml_path': modified_data_yaml_path,
            'class_names': class_names,
            'nc': nc,
            'output_dir': output_base_dir,
            'train_images_path': dataset_paths['train_images'],
            'train_labels_path': dataset_paths['train_labels'],
            'val_images_path': dataset_paths['val_images'],
            'val_labels_path': dataset_paths['val_labels'],
            'test_images_path': dataset_paths['test_images'],
            'test_labels_path': dataset_paths['test_labels']
        }
        
        return dataset_info
        
    except Exception as e:
        print(f"Error processing dataset '{alias}': {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_all_datasets():
    """
    Process all datasets defined in DATASET_CONFIGS.
    
    Returns:
        List of dictionaries containing processed dataset information
    """
    prepared_datasets_info = []
    
    for dataset_config in config.DATASET_CONFIGS:
        dataset_info = process_dataset(dataset_config)
        if dataset_info:
            prepared_datasets_info.append(dataset_info)
    
    print(f"\n{'='*80}")
    print(f"Finished processing {len(prepared_datasets_info)} datasets.")
    print(f"{'='*80}")
    
    return prepared_datasets_info
