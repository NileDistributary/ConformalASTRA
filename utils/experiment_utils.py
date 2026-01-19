"""
Experiment Utilities
Common utilities for experiment scripts with proper encoding handling
Filename: utils/experiment_utils.py
"""
import os
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - won't display plots
import matplotlib.pyplot as plt


def save_results_to_csv(results_dict, experiment_name, append=False):
    """
    Save experiment results to CSV file
    
    Args:
        results_dict: Dictionary with experiment results
        experiment_name: Name of the experiment for filename
        append: If True, append to existing file
    """
    # Create results/csvs directory
    os.makedirs('./results/csvs', exist_ok=True)
    
    # Create filename
    csv_file = f'./results/csvs/{experiment_name}_results.csv'
    
    # Check if file exists
    file_exists = os.path.isfile(csv_file)
    
    # Write to CSV
    mode = 'a' if (append and file_exists) else 'w'
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
        
        # Write header if new file
        if not file_exists or not append:
            writer.writeheader()
        
        writer.writerow(results_dict)
    
    # FIXED: Use ASCII-safe checkmark
    print(f"[OK] Results saved to {csv_file}")


def save_figure(fig, experiment_name, plot_name, timestamp=True):
    """
    Save matplotlib figure without displaying
    
    Args:
        fig: Matplotlib figure object
        experiment_name: Name of the experiment
        plot_name: Name for the plot
        timestamp: If True, add timestamp to filename
    """
    # Create results/figures directory
    os.makedirs('./results/figures', exist_ok=True)
    
    # Create filename
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'./results/figures/{experiment_name}_{plot_name}_{ts}.svg'
    else:
        filename = f'./results/figures/{experiment_name}_{plot_name}.svg'
    
    # Save figure
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    
    # FIXED: Use ASCII-safe checkmark
    print(f"[OK] Figure saved to {filename}")


def save_config(config_dict, experiment_name):
    """
    Save experiment configuration to file
    
    Args:
        config_dict: Dictionary with configuration parameters
        experiment_name: Name of the experiment
    """
    # Create results/configs directory
    os.makedirs('./results/configs', exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = f'./results/configs/{experiment_name}_config_{timestamp}.txt'
    
    # Write configuration
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
    
    # FIXED: Use ASCII-safe checkmark
    print(f"[OK] Config saved to {config_path}")


def print_experiment_header(title):
    """
    Print formatted experiment header
    
    Args:
        title: Experiment title
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: {title}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_experiment_summary(results_dict):
    """
    Print formatted experiment summary
    
    Args:
        results_dict: Dictionary with experiment results
    """
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for key, value in results_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*70 + "\n")