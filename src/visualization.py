"""
Visualization module for Kima Object Detection Training Pipeline

This module handles comparison reporting and visualization.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import config
import paths


def generate_comparison_report(all_results):
    """
    Generate comparison report and visualizations from training results.
    
    Args:
        all_results: List of dictionaries containing training results
        
    Returns:
        pandas DataFrame with comparison results
    """
    if not all_results:
        print("No comparison results found. Please ensure the training and export loop ran successfully.")
        return pd.DataFrame()
    
    # Convert results to DataFrame
    df_comparison = pd.DataFrame(all_results)
    
    if df_comparison.empty:
        print("DataFrame for comparison is empty. No comparison or recommendations can be made.")
        return df_comparison
    
    # Display the comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    csv_path = paths.path_manager.get_comparison_csv_path()
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison results saved to '{csv_path}'")
    
    # Create comparison plots directory
    comparison_plots_dir = paths.path_manager.get_comparison_results_dir()
    comparison_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Comparison plots will be saved to '{comparison_plots_dir}'")
    
    # Generate visualizations
    print("\nGenerating comparative visualizations...")
    
    # Add Model Size column
    if 'Trained Model Path' in df_comparison.columns:
        df_comparison['Model Size (MB)'] = df_comparison['Trained Model Path'].apply(
            lambda x: round(Path(x).stat().st_size / (1024 * 1024), 2) if Path(x).exists() else 0.0
        )
    else:
        df_comparison['Model Size (MB)'] = 0.0
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model Comparison Analysis', fontsize=20, fontweight='bold')
    
    # a. Scatter plot: Accuracy (mAP50) vs. Inference Time (ms)
    ax1 = axes[0, 0]
    size_scale_factor = 20  # Adjust this factor as needed
    for idx, row in df_comparison.iterrows():
        ax1.scatter(
            row['Inference Time (ms)'], 
            row['mAP50'], 
            s=row['Model Size (MB)'] * size_scale_factor, 
            alpha=0.7, 
            label=row['model_architecture']
        )
    ax1.set_xlabel('Inference Time (ms)', fontsize=12)
    ax1.set_ylabel('mAP50', fontsize=12)
    ax1.set_title('Accuracy (mAP50) vs. Speed (Inference Time)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(title='Model', loc='best', fontsize=10)
    
    # b. Bar chart: mAP50
    ax2 = axes[0, 1]
    df_sorted_map = df_comparison.sort_values(by='mAP50', ascending=True, inplace=False)
    df_sorted_map.plot(x='model_architecture', y='mAP50', kind='barh', ax=ax2, legend=False, color='skyblue')
    ax2.set_xlabel('mAP50', fontsize=12)
    ax2.set_ylabel('Model Architecture', fontsize=12)
    ax2.set_title('Model Accuracy (mAP50)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(df_sorted_map['mAP50']):
        ax2.text(v + 0.01, i, str(v), color='black', va='center')
    
    # c. Bar chart: Inference Time (ms)
    ax3 = axes[1, 0]
    df_sorted_time = df_comparison.sort_values(by='Inference Time (ms)', ascending=False, inplace=False)
    df_sorted_time.plot(x='model_architecture', y='Inference Time (ms)', kind='barh', ax=ax3, legend=False, color='lightcoral')
    ax3.set_xlabel('Inference Time (ms)', fontsize=12)
    ax3.set_ylabel('Model Architecture', fontsize=12)
    ax3.set_title('Inference Speed', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(df_sorted_time['Inference Time (ms)']):
        ax3.text(v + 0.1, i, str(v), color='black', va='center')
    
    # d. Bar chart: Model Size (MB)
    ax4 = axes[1, 1]
    df_sorted_size = df_comparison.sort_values(by='Model Size (MB)', ascending=True, inplace=False)
    df_sorted_size.plot(x='model_architecture', y='Model Size (MB)', kind='barh', ax=ax4, legend=False, color='lightgreen')
    ax4.set_xlabel('Size (MB)', fontsize=12)
    ax4.set_ylabel('Model Architecture', fontsize=12)
    ax4.set_title('Model Size', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(df_sorted_size['Model Size (MB)']):
        ax4.text(v + 0.5, i, str(v), color='black', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = paths.path_manager.get_comparison_plot_path()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"✓ Comparison visualizations saved to '{plot_path}'")
    
    # Provide recommendations
    print_recommendations(df_comparison)
    
    return df_comparison


def print_recommendations(df_comparison):
    """
    Print model recommendations based on various criteria.
    
    Args:
        df_comparison: DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("BEST MODEL RECOMMENDATIONS")
    print("="*80)
    
    # a. Best accuracy (mAP50)
    best_map50 = df_comparison.loc[df_comparison['mAP50'].idxmax()]
    print(f"\n🏆 Best Accuracy (mAP50): {best_map50['model_architecture']} (Dataset: {best_map50['dataset_alias']})")
    print(f"   mAP50: {best_map50['mAP50']:.4f}")
    print(f"   Inference: {best_map50['Inference Time (ms)']:.2f} ms")
    print(f"   Size: {best_map50['Model Size (MB)']:.2f} MB")
    
    # b. Fastest inference
    fastest = df_comparison.loc[df_comparison['Inference Time (ms)'].idxmin()]
    print(f"\n⚡ Fastest Inference: {fastest['model_architecture']} (Dataset: {fastest['dataset_alias']})")
    print(f"   Inference: {fastest['Inference Time (ms)']:.2f} ms")
    print(f"   mAP50: {fastest['mAP50']:.4f}")
    print(f"   Size: {fastest['Model Size (MB)']:.2f} MB")
    
    # c. Best balance (mAP50 / (Inference Time (ms) + 1))
    df_comparison['Balance Score'] = df_comparison['mAP50'] / (df_comparison['Inference Time (ms)'] + 1e-6)
    best_balance = df_comparison.loc[df_comparison['Balance Score'].idxmax()]
    print(f"\n⚖️  Best Balance (Accuracy/Speed): {best_balance['model_architecture']} (Dataset: {best_balance['dataset_alias']})")
    print(f"   Balance Score: {best_balance['Balance Score']:.4f}")
    print(f"   mAP50: {best_balance['mAP50']:.4f}")
    print(f"   Inference: {best_balance['Inference Time (ms)']:.2f} ms")
    print(f"   Size: {best_balance['Model Size (MB)']:.2f} MB")
    
    # d. Smallest model
    smallest = df_comparison.loc[df_comparison['Model Size (MB)'].idxmin()]
    print(f"\n📦 Smallest Model: {smallest['model_architecture']} (Dataset: {smallest['dataset_alias']})")
    print(f"   Size: {smallest['Model Size (MB)']:.2f} MB")
    print(f"   mAP50: {smallest['mAP50']:.4f}")
    print(f"   Inference: {smallest['Inference Time (ms)']:.2f} ms")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION FOR REAL-TIME MOBILE DETECTION")
    print("="*80)
    
    # e. Final recommendation for real-time mobile detection
    good_accuracy_models = df_comparison[df_comparison['mAP50'] > 0.2].copy()
    if not good_accuracy_models.empty:
        recommended_model = good_accuracy_models.loc[good_accuracy_models['Inference Time (ms)'].idxmin()]
        print(f"\n✅ RECOMMENDED: {recommended_model['model_architecture']} on {recommended_model['dataset_alias']}")
        print(f"   Reason: Best speed with acceptable accuracy for real-time mobile applications.")
        print(f"   mAP50: {recommended_model['mAP50']:.4f}")
        print(f"   Inference Time: {recommended_model['Inference Time (ms)']:.2f} ms")
        print(f"   Model Size: {recommended_model['Model Size (MB)']:.2f} MB")
        print(f"   Trained Model Path: {recommended_model['Trained Model Path']}")
    else:
        print("No models achieved sufficient accuracy (mAP50 > 0.2) to provide a specific real-time mobile recommendation.")
        print("Please review training parameters or dataset quality.")
    print("="*80)
