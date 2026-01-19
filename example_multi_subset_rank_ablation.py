"""
Example: Multi-Subset Rank Ablation Experiment

This script demonstrates how to adapt your existing experiment_ablation_rank.py
to run on multiple subsets systematically.

Key changes from single-subset version:
1. Uses subset_utils for configuration management
2. Loops over specified subsets
3. Organizes results by subset
4. Generates cross-subset comparison plots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import munch
from data.eth import ETH_dataset
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.misc import set_seed
from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI
from astra_wrapper import ASTRASklearnWrapper
import time
import pandas as pd
from datetime import datetime

# Import subset utilities
try:
    from subset_utils import (
        get_subset_config,
        get_available_subsets, 
        load_subset_yaml_config,
        get_results_subdir,
        print_subset_info
    )
except ImportError:
    # If running from root directory
    from utils.subset_utils import (
        get_subset_config,
        get_available_subsets,
        load_subset_yaml_config,
        get_results_subdir,
        print_subset_info
    )


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(cfg, subset, split_ratio=0.90):
    """Load and prepare data for a specific subset"""
    reshape_size = cfg.DATA.MIN_RESHAPE_SIZE
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    
    transforms = A.Compose([
        A.LongestMaxSize(reshape_size),
        A.PadIfNeeded(reshape_size, reshape_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean, std, max_pixel_value=255.0),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='yx'))
    
    # Ensure subset is set correctly
    cfg.SUBSET = subset
    full_dataset = ETH_dataset(cfg, mode='testing', img_transforms=transforms)
    
    loader = data.DataLoader(full_dataset, batch_size=1, shuffle=False, 
                            pin_memory=True, drop_last=False)
    
    all_past_trajectories = []
    all_future_trajectories = []
    all_images = []
    all_num_valid = []
    
    for batch in loader:
        past_loc, fut_loc, num_valid, imgs, gt_maps, traj_coords = batch
        all_past_trajectories.append(past_loc.numpy())
        all_future_trajectories.append(fut_loc.numpy())
        all_images.append(imgs.numpy())
        all_num_valid.append(num_valid.numpy())
    
    all_past_trajectories = np.concatenate(all_past_trajectories, axis=0)
    all_future_trajectories = np.concatenate(all_future_trajectories, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_num_valid = np.concatenate(all_num_valid, axis=0)
    
    # Split into calibration and test
    n_total = len(all_past_trajectories)
    n_train = int(n_total * split_ratio)
    
    X_train = (all_past_trajectories[:n_train], all_images[:n_train], all_num_valid[:n_train])
    Y_train = all_future_trajectories[:n_train]
    X_test = (all_past_trajectories[n_train:], all_images[n_train:], all_num_valid[n_train:])
    Y_test = all_future_trajectories[n_train:]
    
    print(f"  Data split: {n_train} calibration, {len(Y_test)} test samples")
    
    return X_train, Y_train, X_test, Y_test


# ============================================================================
# EXPERIMENT LOGIC
# ============================================================================

def run_rank_experiment_on_subset(subset_name, ranks_to_test, alpha=0.1, device='cuda'):
    """
    Run rank ablation experiment on a single subset.
    
    Args:
        subset_name: Name of the subset to process
        ranks_to_test: List of rank values to test
        alpha: Significance level
        device: Device to use
        
    Returns:
        list: Results for all ranks
    """
    print(f"\n{'='*70}")
    print(f"SUBSET: {subset_name}")
    print(f"{'='*70}\n")
    
    # Get subset configuration
    subset_config = get_subset_config(subset_name)
    print_subset_info(subset_name)
    
    # Load YAML config
    cfg = load_subset_yaml_config(subset_name)
    cfg.device = device
    cfg.device_list = [device] if device != 'cpu' else []
    
    # Prepare data
    print("Loading data...")
    X_train, Y_train, X_test, Y_test = prepare_data(cfg, subset_name)
    
    # Initialize ASTRA
    print("\nInitializing ASTRA...")
    astra_wrapper = ASTRASklearnWrapper(
        config_path=subset_config['config_path'],
        pretrained_weights_path=subset_config['pretrained_weights'],
        unet_weights_path=subset_config['unet_weights'],
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    astra_wrapper.fit(X_train, Y_train)
    print("✓ ASTRA initialized")
    
    # Run experiments for each rank
    subset_results = []
    
    for rank in ranks_to_test:
        rank_label = 'Full' if rank is None else f'{rank}'
        print(f"\n--- Testing Rank: {rank_label} ---")
        
        start_time = time.time()
        
        try:
            # Initialize SPCI
            spci = SPCI_and_EnbPI(
                astra_wrapper,
                X_train, Y_train,
                alpha=alpha,
                stride=1,
                past_window=100,
                method='SPCI',
                rank_cov=rank
            )
            
            # Get predictions and coverage
            lower, upper = spci.fit_bootstrap_agg(X_test)
            coverage = spci.compute_coverage(Y_test, lower, upper)
            volume = spci.compute_volume(lower, upper)
            
            elapsed = time.time() - start_time
            
            result = {
                'subset': subset_name,
                'rank': rank_label,
                'rank_value': rank,
                'coverage': coverage,
                'volume': volume,
                'time': elapsed,
                'num_test': len(Y_test),
                'target_coverage': 1 - alpha
            }
            
            subset_results.append(result)
            
            print(f"  Coverage: {coverage*100:.2f}%")
            print(f"  Volume: {volume:.2e}")
            print(f"  Time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            result = {
                'subset': subset_name,
                'rank': rank_label,
                'rank_value': rank,
                'coverage': None,
                'volume': None,
                'time': time.time() - start_time,
                'num_test': len(Y_test),
                'target_coverage': 1 - alpha,
                'error': str(e)
            }
            subset_results.append(result)
    
    return subset_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cross_subset_comparison(all_results_df, metric='coverage'):
    """
    Create comparison plots across subsets.
    
    Args:
        all_results_df: DataFrame with all results
        metric: Metric to plot ('coverage' or 'volume')
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    subsets = all_results_df['subset'].unique()
    ranks = all_results_df['rank'].unique()
    
    # Plot lines for each subset
    for subset in subsets:
        subset_data = all_results_df[all_results_df['subset'] == subset]
        
        if metric == 'coverage':
            values = subset_data['coverage'] * 100
            ylabel = 'Coverage (%)'
            title = 'Coverage vs Rank Across Subsets'
        else:
            values = subset_data['volume']
            ylabel = 'Volume'
            title = 'Volume vs Rank Across Subsets'
            ax.set_yscale('log')
        
        ax.plot(range(len(ranks)), values, 
               marker='o', linewidth=2, markersize=8, label=subset)
    
    # Add target coverage line if plotting coverage
    if metric == 'coverage':
        target = all_results_df['target_coverage'].iloc[0] * 100
        ax.axhline(y=target, color='red', linestyle='--', 
                  linewidth=2, label=f'Target ({target:.0f}%)')
    
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run rank ablation across multiple subsets'
    )
    parser.add_argument('--subsets', nargs='+', default=['all'],
                       help='Subsets to process (default: all available)')
    parser.add_argument('--ranks', nargs='+', type=int, default=[4, 8, 12, 16, 20],
                       help='Rank values to test')
    parser.add_argument('--full-rank', action='store_true',
                       help='Also test full rank (None)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Significance level')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Determine which subsets to process
    if 'all' in args.subsets:
        subsets_to_process = get_available_subsets()
        print(f"\nProcessing all available subsets: {', '.join(subsets_to_process)}")
    else:
        subsets_to_process = args.subsets
    
    if len(subsets_to_process) == 0:
        print("Error: No subsets available to process")
        return
    
    # Prepare ranks to test
    ranks_to_test = args.ranks
    if args.full_rank:
        ranks_to_test.append(None)
    
    print(f"\nRanks to test: {ranks_to_test}")
    print(f"Alpha: {args.alpha}")
    print(f"Device: {args.device}")
    
    # Set random seed
    set_seed(42)
    
    # Run experiments on all subsets
    all_results = []
    
    for subset in subsets_to_process:
        try:
            subset_results = run_rank_experiment_on_subset(
                subset_name=subset,
                ranks_to_test=ranks_to_test,
                alpha=args.alpha,
                device=args.device
            )
            all_results.extend(subset_results)
            
        except Exception as e:
            print(f"\n✗ Failed to process {subset}: {str(e)}")
            continue
    
    # Save results
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/multi_subset_rank_ablation'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        csv_path = os.path.join(output_dir, f'rank_ablation_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results to: {csv_path}")
        
        # Generate comparison plots
        if len(df['subset'].unique()) > 1:
            # Coverage comparison
            fig_coverage = plot_cross_subset_comparison(df, metric='coverage')
            fig_path = os.path.join(output_dir, f'coverage_comparison_{timestamp}.svg')
            fig_coverage.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_coverage)
            print(f"✓ Saved coverage plot: {fig_path}")
            
            # Volume comparison
            fig_volume = plot_cross_subset_comparison(df, metric='volume')
            fig_path = os.path.join(output_dir, f'volume_comparison_{timestamp}.svg')
            fig_volume.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig_volume)
            print(f"✓ Saved volume plot: {fig_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL SUBSETS")
        print(f"{'='*70}")
        
        for subset in df['subset'].unique():
            print(f"\n{subset}:")
            subset_df = df[df['subset'] == subset]
            for _, row in subset_df.iterrows():
                print(f"  Rank {row['rank']}: "
                      f"Coverage={row['coverage']*100:.2f}%, "
                      f"Volume={row['volume']:.2e}")
        
        print(f"\n{'='*70}\n")
    else:
        print("\n✗ No results to save")


if __name__ == "__main__":
    main()
