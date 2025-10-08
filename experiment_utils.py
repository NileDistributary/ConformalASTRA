"""
Utilities for running and saving experiments
"""
import os
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Create results directory structure
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "csvs").mkdir(exist_ok=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True)
(RESULTS_DIR / "models").mkdir(exist_ok=True)


def get_timestamp():
    """Get formatted timestamp for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results_to_csv(results_dict, experiment_name, append=True):
    """
    Save experiment results to CSV file
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results (must have consistent keys for appending)
    experiment_name : str
        Name of experiment (e.g., 'baseline_comparison')
    append : bool
        If True, append to existing CSV. If False, create new file with timestamp
    """
    csv_path = RESULTS_DIR / "csvs" / f"{experiment_name}.csv"
    
    # Convert dict to DataFrame
    df = pd.DataFrame([results_dict])
    
    # Add timestamp column
    df['timestamp'] = get_timestamp()
    
    if append and csv_path.exists():
        # Append to existing file
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"✓ Results appended to {csv_path}")
    else:
        # Create new file
        df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"✓ Results saved to {csv_path}")
    
    return csv_path


def save_figure(fig, experiment_name, description, timestamp=True):
    """
    Save matplotlib figure with proper naming
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    experiment_name : str
        Name of experiment
    description : str
        Description of what the figure shows (e.g., 'coverage_vs_alpha')
    timestamp : bool
        Whether to add timestamp to filename
    """
    if timestamp:
        filename = f"{experiment_name}_{description}_{get_timestamp()}.png"
    else:
        filename = f"{experiment_name}_{description}.png"
    
    fig_path = RESULTS_DIR / "figures" / filename
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to {fig_path}")
    
    return fig_path


def save_config(config_dict, experiment_name):
    """Save experiment configuration as JSON"""
    config_path = RESULTS_DIR / "csvs" / f"{experiment_name}_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"✓ Config saved to {config_path}")
    return config_path


def load_results_csv(experiment_name):
    """Load results CSV as DataFrame"""
    csv_path = RESULTS_DIR / "csvs" / f"{experiment_name}.csv"
    
    if not csv_path.exists():
        print(f"⚠ No results file found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} results from {csv_path}")
    return df


def print_experiment_header(experiment_name):
    """Print formatted experiment header"""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name.upper()}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def print_experiment_summary(results_dict):
    """Print formatted summary of results"""
    print("\n" + "-"*70)
    print("RESULTS SUMMARY:")
    print("-"*70)
    for key, value in results_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("-"*70 + "\n")