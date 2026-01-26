"""
Path management module for Kima Object Detection Training Pipeline

This module handles all file paths and directory structures.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class PathManager:
    """Manages all paths for the project"""
    
    def __init__(self, base_dir=None):
        """
        Initialize PathManager
        
        Args:
            base_dir: Base directory for the project. If None, uses project root
        """
        if base_dir is None:
            # Use the project root (parent of src/)
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        
        # Data directory for all outputs, datasets, and runs
        self.data_dir = self.base_dir / 'data'
        
        # Main directories within data/
        self.outputs_dir = self.data_dir / 'outputs'
        self.datasets_dir = self.data_dir / 'datasets'
        self.runs_dir = self.data_dir / 'runs' / 'all_trainings'
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_output_dir(self, dataset_alias):
        """
        Get output directory for a specific dataset
        
        Args:
            dataset_alias: Alias of the dataset
            
        Returns:
            Path to dataset output directory
        """
        return self.outputs_dir / dataset_alias
    
    def get_dataset_analysis_dir(self, dataset_alias):
        """
        Get analysis output directory for a specific dataset
        
        Args:
            dataset_alias: Alias of the dataset
            
        Returns:
            Path to dataset analysis directory
        """
        return self.get_dataset_output_dir(dataset_alias) / 'dataset_analysis'
    
    def get_model_output_dir(self, model_arch, dataset_alias):
        """
        Get output directory for a specific model-dataset combination
        
        Args:
            model_arch: Model architecture name
            dataset_alias: Alias of the dataset
            
        Returns:
            Path to model output directory
        """
        run_name = f"{model_arch}_{dataset_alias}"
        return self.runs_dir / run_name
    
    def get_visual_tests_dir(self, model_arch, dataset_alias):
        """
        Get visual tests directory for a specific model-dataset combination
        
        Args:
            model_arch: Model architecture name
            dataset_alias: Alias of the dataset
            
        Returns:
            Path to visual tests directory
        """
        return self.get_model_output_dir(model_arch, dataset_alias) / 'visual_tests'
    
    def get_comparison_results_dir(self):
        """
        Get comparison results directory
        
        Returns:
            Path to comparison results directory
        """
        return self.outputs_dir / 'comparison_results'
    
    def get_models_dir(self):
        """
        Get models output directory
        
        Returns:
            Path to models directory
        """
        return self.outputs_dir / 'models'
    
    def get_comparison_csv_path(self):
        """
        Get path to comparison CSV file
        
        Returns:
            Path to comparison CSV file
        """
        return self.outputs_dir / 'model_comparison_results.csv'
    
    def get_comparison_plot_path(self):
        """
        Get path to comparison plot
        
        Returns:
            Path to comparison plot file
        """
        return self.get_comparison_results_dir() / 'model_comparison_summary.png'
    
    def get_zip_archive_path(self):
        """
        Get path to zip archive of outputs
        
        Returns:
            Path to zip archive file
        """
        return self.data_dir / 'outputs.zip'


# Global path manager instance
path_manager = PathManager()
