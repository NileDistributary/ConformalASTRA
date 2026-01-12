"""
Utilities for Multi-Subset Experiments

This module provides helper functions to easily adapt existing experiments
to run across multiple ETH-UCY subsets.

Usage in your experiment scripts:
    from utils.subset_utils import get_subset_config, get_all_subsets
    
    # Get configuration for a specific subset
    config = get_subset_config('hotel')
    
    # Run experiment on all available subsets
    for subset_name, subset_config in get_all_subsets():
        # Your experiment code here
"""

import os
from pathlib import Path
import yaml
import munch


# ============================================================================
# SUBSET CONFIGURATION
# ============================================================================

SUBSET_CONFIGS = {
    'eth': {
        'config_path': 'configs/eth.yaml',
        'pretrained_weights': 'pretrained_astra_weights/eth_best_model.pth',
        'unet_weights': 'pretrained_unet_weights/eth_unet_model_best.pt',
        'description': 'ETH Pedestrian Dataset',
        'expected_ade': 0.47,  # From ASTRA paper
        'expected_fde': 0.82
    },
    'hotel': {
        'config_path': 'configs/hotel.yaml',
        'pretrained_weights': 'pretrained_astra_weights/hotel_best_model.pth',
        'unet_weights': 'pretrained_unet_weights/eth_unet_model_best.pt',  # Shared U-Net
        'description': 'Hotel Scene',
        'expected_ade': 0.29,
        'expected_fde': 0.56
    },
    'univ': {
        'config_path': 'configs/univ.yaml',
        'pretrained_weights': 'pretrained_astra_weights/univ_best_model.pth',
        'unet_weights': 'pretrained_unet_weights/eth_unet_model_best.pt',
        'description': 'University Scene',
        'expected_ade': 0.55,
        'expected_fde': 1.00
    },
    'zara01': {
        'config_path': 'configs/zara01.yaml',
        'pretrained_weights': 'pretrained_astra_weights/zara01_best_model.pth',
        'unet_weights': 'pretrained_unet_weights/eth_unet_model_best.pt',
        'description': 'Zara 01 Scene',
        'expected_ade': 0.34,
        'expected_fde': 0.71
    },
    'zara02': {
        'config_path': 'configs/zara02.yaml',
        'pretrained_weights': 'pretrained_astra_weights/zara02_best_model.pth',
        'unet_weights': 'pretrained_unet_weights/eth_unet_model_best.pt',
        'description': 'Zara 02 Scene',
        'expected_ade': 0.24,
        'expected_fde': 0.41
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_subset_config(subset_name):
    """
    Get configuration for a specific subset.
    
    Args:
        subset_name: Name of the subset ('eth', 'hotel', 'univ', 'zara01', 'zara02')
        
    Returns:
        dict: Configuration dictionary with paths and metadata
        
    Raises:
        ValueError: If subset_name is not recognized
    """
    if subset_name not in SUBSET_CONFIGS:
        available = ', '.join(SUBSET_CONFIGS.keys())
        raise ValueError(f"Unknown subset '{subset_name}'. Available: {available}")
    
    return SUBSET_CONFIGS[subset_name].copy()


def check_subset_files(subset_name):
    """
    Check if all required files exist for a subset.
    
    Args:
        subset_name: Name of the subset
        
    Returns:
        tuple: (bool, list) - (all_exist, missing_files)
    """
    config = get_subset_config(subset_name)
    
    required_files = [
        config['config_path'],
        config['pretrained_weights'],
        config['unet_weights']
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    return len(missing) == 0, missing


def get_available_subsets():
    """
    Get list of subsets that have all required files.
    
    Returns:
        list: Names of subsets that are ready to use
    """
    available = []
    for subset_name in SUBSET_CONFIGS.keys():
        files_exist, _ = check_subset_files(subset_name)
        if files_exist:
            available.append(subset_name)
    return available


def get_all_subsets():
    """
    Generator that yields (subset_name, config) for all configured subsets.
    
    Yields:
        tuple: (subset_name, config_dict)
    """
    for subset_name, config in SUBSET_CONFIGS.items():
        yield subset_name, config.copy()


def load_subset_yaml_config(subset_name):
    """
    Load the YAML config file for a subset and return as munch object.
    
    Args:
        subset_name: Name of the subset
        
    Returns:
        munch.Munch: Configuration object
    """
    config = get_subset_config(subset_name)
    config_path = config['config_path']
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg = munch.munchify(cfg)
    
    # Ensure SUBSET field matches
    cfg.SUBSET = subset_name
    
    return cfg


def get_results_subdir(subset_name, experiment_type):
    """
    Get the results subdirectory path for a specific subset and experiment.
    
    Args:
        subset_name: Name of the subset
        experiment_type: Type of experiment (e.g., 'ablation_rank', 'main_results')
        
    Returns:
        str: Path to results subdirectory
    """
    subdir = f"results/{experiment_type}/{subset_name}"
    os.makedirs(subdir, exist_ok=True)
    return subdir


def print_subset_info(subset_name):
    """
    Print information about a subset.
    
    Args:
        subset_name: Name of the subset
    """
    config = get_subset_config(subset_name)
    files_exist, missing = check_subset_files(subset_name)
    
    print(f"\n{'='*60}")
    print(f"Subset: {subset_name}")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Config: {config['config_path']}")
    print(f"Weights: {config['pretrained_weights']}")
    print(f"U-Net: {config['unet_weights']}")
    print(f"Expected ADE: {config['expected_ade']}")
    print(f"Expected FDE: {config['expected_fde']}")
    
    if files_exist:
        print(f"Status: ✓ All files present")
    else:
        print(f"Status: ✗ Missing files:")
        for f in missing:
            print(f"  - {f}")
    print(f"{'='*60}\n")


def print_all_subsets_status():
    """Print status of all configured subsets."""
    print("\n" + "="*70)
    print("ETH-UCY SUBSETS STATUS")
    print("="*70)
    
    for subset_name in SUBSET_CONFIGS.keys():
        files_exist, missing = check_subset_files(subset_name)
        config = get_subset_config(subset_name)
        
        status = "✓" if files_exist else "✗"
        print(f"\n{status} {subset_name} ({config['description']})")
        
        if not files_exist:
            print(f"  Missing files:")
            for f in missing:
                print(f"    - {f}")
    
    print("\n" + "="*70)


# ============================================================================
# EXPERIMENT TEMPLATE
# ============================================================================

def subset_experiment_template():
    """
    Template showing how to structure an experiment to run on multiple subsets.
    This is a code example, not meant to be executed directly.
    """
    
    # Example 1: Run on a specific subset
    def run_single_subset():
        subset_name = 'hotel'
        config = get_subset_config(subset_name)
        
        # Check files exist
        files_exist, missing = check_subset_files(subset_name)
        if not files_exist:
            print(f"Cannot run on {subset_name}, missing: {missing}")
            return
        
        # Load YAML config
        cfg = load_subset_yaml_config(subset_name)
        
        # Initialize your ASTRA wrapper
        from astra_wrapper import ASTRASklearnWrapper
        astra = ASTRASklearnWrapper(
            config_path=config['config_path'],
            pretrained_weights_path=config['pretrained_weights'],
            unet_weights_path=config['unet_weights'],
            use_pretrained_unet=True,
            device='cuda',
            dataset='ETH_UCY'
        )
        
        # Run your experiment...
        
        # Save results to subset-specific directory
        results_dir = get_results_subdir(subset_name, 'my_experiment')
        # Save your results to results_dir
    
    # Example 2: Run on all available subsets
    def run_all_subsets():
        all_results = []
        
        for subset_name in get_available_subsets():
            print(f"\n{'='*60}")
            print(f"Processing {subset_name}")
            print(f"{'='*60}")
            
            config = get_subset_config(subset_name)
            cfg = load_subset_yaml_config(subset_name)
            
            # Initialize ASTRA for this subset
            from astra_wrapper import ASTRASklearnWrapper
            astra = ASTRASklearnWrapper(
                config_path=config['config_path'],
                pretrained_weights_path=config['pretrained_weights'],
                unet_weights_path=config['unet_weights'],
                use_pretrained_unet=True,
                device='cuda',
                dataset='ETH_UCY'
            )
            
            # Run experiment
            result = {
                'subset': subset_name,
                # ... your metrics
            }
            all_results.append(result)
            
            # Save subset-specific results
            results_dir = get_results_subdir(subset_name, 'my_experiment')
            # Save to results_dir
        
        # Save combined results
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv('results/all_subsets_summary.csv', index=False)


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Subset configuration utilities')
    parser.add_argument('--status', action='store_true',
                       help='Print status of all subsets')
    parser.add_argument('--info', type=str, metavar='SUBSET',
                       help='Print detailed info for a specific subset')
    parser.add_argument('--available', action='store_true',
                       help='List subsets that are ready to use')
    
    args = parser.parse_args()
    
    if args.status:
        print_all_subsets_status()
    elif args.info:
        print_subset_info(args.info)
    elif args.available:
        available = get_available_subsets()
        print(f"\nAvailable subsets: {', '.join(available)}")
        print(f"Total: {len(available)}/{len(SUBSET_CONFIGS)}\n")
    else:
        parser.print_help()
