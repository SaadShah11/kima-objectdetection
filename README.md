# Kima Object Detection Training Pipeline

A modular Python pipeline for training and comparing YOLO models for real-time basketball object detection, optimized for mobile deployment.

## Project Structure

```
Kima-Main/
├── main.py                 # Main entry point - run this to start the pipeline
├── config.py               # Configuration settings (datasets, models, hyperparameters)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── paths.py           # Path management and directory structure
│   ├── utils.py            # Utility functions (class counts, bbox plotting, etc.)
│   ├── data_processing.py  # Dataset download, preparation, and analysis
│   ├── model_training.py   # Model training, validation, and export
│   ├── visualization.py   # Comparison reporting and visualization
│   └── archiving.py        # Output archiving and model file organization
└── data/                   # Data directory (created automatically)
    ├── outputs/            # All output files (models, plots, reports)
    ├── datasets/           # Downloaded datasets
    └── runs/               # Training runs
```

## Features

- **Automated Dataset Processing**: Download datasets from Roboflow, correct YAML files, and perform analysis
- **Multi-Model Training**: Train multiple YOLO architectures (yolo11n, yolo11s, etc.) on different datasets
- **Mobile Export**: Export trained models to CoreML (iOS) and TFLite (Android) formats
- **Comprehensive Comparison**: Generate comparison reports, visualizations, and recommendations
- **Organized Outputs**: All results are saved in a structured `data/outputs/` directory

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU Support (Recommended for Faster Training):**
   
   The pipeline automatically detects and uses GPU if available, falling back to CPU if not. To enable GPU acceleration:
   
   **For NVIDIA GPUs (CUDA):**
   
   First, verify your GPU is detected:
   ```bash
   nvidia-smi
   ```
   
   Then install PyTorch with CUDA support:
   ```bash
   # For CUDA 12.4 (recommended for newer GPUs)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   
   # For CUDA 11.8 (if CUDA 12.4 doesn't work)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   **Verify GPU is detected:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```
   
   You should see `CUDA available: True` and your GPU name if installation was successful.
   
   **Note:** If you already have PyTorch installed (CPU-only version), uninstall it first:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```
   Then install the CUDA version as shown above.
   
   **Device Selection:**
   - The pipeline uses `device: 'auto'` by default, which automatically uses GPU if available
   - You can force CPU by setting `device: 'cpu'` in `config.py`
   - You can specify a specific GPU by setting `device: 0` (for GPU 0), `device: 1` (for GPU 1), etc.

3. **On Linux systems, install system dependencies (optional, for AVIF support):**
   ```bash
   sudo apt-get update && sudo apt-get install -y libavif-dev libheif-dev
   ```

## Configuration

Edit `config.py` to configure:

- **Dataset Configurations**: Add your Roboflow API key and dataset details
- **Model Architectures**: Choose which YOLO models to train
- **Training Hyperparameters**: Adjust epochs, batch size, image size, etc.
- **Workflow Flags**: Enable/disable specific features

### Example Configuration

```python
DATASET_CONFIGS = [
    {
        'ROBOFLOW_API_KEY': os.getenv('ROBOFLOW_API_KEY', 'your_key_here'),
        'WORKSPACE': 'your-workspace',
        'PROJECT': 'your-project',
        'VERSION': 2,
        'ALIAS': 'my_dataset'
    },
]

MODEL_ARCHITECTURES = ['yolo11n', 'yolo11s']

TRAINING_HYPERPARAMETERS = {
    'epochs': 60,
    'imgsz': 416,
    'batch': 16,
    'patience': 10,
    'device': 'auto',  # 'auto' (use GPU if available, else CPU), 'cpu', or GPU index (0, 1, 2, etc.)
}
```

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Download and prepare all datasets
2. Analyze dataset characteristics
3. Train all specified models
4. Export models to mobile formats
5. Generate comparison reports
6. Archive all outputs

### Run Individual Modules

You can also import and use individual modules:

```python
from src import data_processing
from src import model_training
from src import visualization

# Prepare datasets
datasets = data_processing.prepare_all_datasets()

# Train models
results = model_training.train_all_models(datasets)

# Generate comparison
df = visualization.generate_comparison_report(results)
```

## Output Structure

After running the pipeline, you'll find:

```
data/
├── outputs/
│   ├── all_trainings/              # Training runs for each model-dataset combination
│   ├── comparison_results/        # Comparison visualizations
│   ├── models/                     # Exported CoreML and TFLite models
│   ├── {dataset_alias}/           # Dataset-specific outputs
│   │   └── dataset_analysis/       # Dataset analysis plots and samples
│   ├── model_comparison_results.csv  # Comparison metrics in CSV format
│   └── outputs.zip                 # Archived outputs
├── datasets/                       # Downloaded datasets from Roboflow
└── runs/                           # Training run outputs
```

## Model Export Formats

- **CoreML (.mlpackage)**: For iOS deployment
  - FP16 precision
  - Built-in NMS
  - Optimized for Apple devices

- **TFLite (.tflite)**: For Android deployment
  - INT8 quantization
  - Built-in NMS
  - Optimized for mobile inference

## Performance Tips

### Training Performance
- **Use GPU for training:** GPU acceleration can be 10-50x faster than CPU training
- **Batch size:** Larger batch sizes (16-32) work better with GPU, smaller (4-8) for CPU
- **Image size:** 416x416 is faster than 640x640 with minimal accuracy loss
- **Model selection:** `yolo11n` trains faster than `yolo11s` with slightly lower accuracy

### Inference Performance
- Use image size 416x416 for fastest inference (vs 640x640)
- Process every 2nd or 3rd frame if needed for 30+ FPS
- Use GPU acceleration when available on device
- Adjust confidence threshold based on your use case (default: 0.25)

## Troubleshooting

### GPU Not Detected

**Symptoms:** Pipeline shows "CUDA available: False" or "Device Type: CPU"

**Solutions:**
1. **Verify GPU is detected by system:**
   ```bash
   nvidia-smi
   ```
   If this doesn't work, ensure NVIDIA drivers are installed.

2. **Check PyTorch CUDA installation:**
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```
   
3. **If PyTorch shows CPU-only version:**
   - Uninstall: `pip uninstall torch torchvision torchaudio -y`
   - Reinstall with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
   
4. **If GPU still not detected:**
   - Ensure CUDA driver version is compatible (check with `nvidia-smi`)
   - Try CUDA 11.8 instead: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - As a fallback, set `device: 'cpu'` in `config.py` to use CPU training

### Out of Memory (OOM) Errors

**Symptoms:** Training crashes with CUDA out of memory errors

**Solutions:**
- Reduce `batch` size in `TRAINING_HYPERPARAMETERS` (e.g., from 16 to 8 or 4)
- Reduce `imgsz` (image size) from 640 to 416
- Close other applications using GPU memory
- Use a smaller model architecture (e.g., `yolo11n` instead of `yolo11s`)

### Roboflow API Key Issues
Set the `ROBOFLOW_API_KEY` environment variable:
```bash
export ROBOFLOW_API_KEY='your_key_here'
```

Or update it directly in `config.py` (less secure)

## License

This project uses Ultralytics YOLO, which is licensed under AGPL-3.0 License.

## Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [Roboflow Documentation](https://docs.roboflow.com/)
