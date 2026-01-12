"""
Quick Baseline Comparison - Hotel Subset Only

Simple script to test baseline comparison (Coordinate-wise vs MultiDimSPCI) on hotel subset.
Designed for quick deployment and testing.

Usage:
    set CUDA_VISIBLE_DEVICES=-1
    python baseline_hotel_quick.py
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
from baseline_coordinatewise import CoordinateWiseCP
import time
import pandas as pd
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

# Fixed parameters for quick test
CONFIG_PATH = 'configs/hotel.yaml'
SUBSET = 'hotel'
ALPHA = 0.1  # 90% confidence
SPLIT_RATIO = 0.90  # 90% calibration, 10% test
DEVICE = 'cpu'  # Change to 'cuda' if you have GPU


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(cfg, subset, split_ratio=0.90):
    """Load and prepare data"""
    print("\nLoading data...")
    
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
    
    n_samples = len(all_past_trajectories)
    n_train = int(n_samples * split_ratio)
    
    X_train = {
        'past_trajectories': all_past_trajectories[:n_train],
        'images': all_images[:n_train],
        'num_valid': all_num_valid[:n_train]
    }
    Y_train = all_future_trajectories[:n_train].reshape(n_train, -1)
    
    X_test = {
        'past_trajectories': all_past_trajectories[n_train:],
        'images': all_images[n_train:],
        'num_valid': all_num_valid[n_train:]
    }
    Y_test = all_future_trajectories[n_train:].reshape(n_samples - n_train, -1)
    
    print(f"✓ Loaded {n_samples} samples")
    print(f"  - Calibration: {n_train} samples")
    print(f"  - Test: {len(Y_test)} samples")
    
    return X_train, Y_train, X_test, Y_test


# ============================================================================
# BASELINE METHODS
# ============================================================================

def run_coordinatewise(astra_wrapper, X_train, Y_train, X_test, Y_test, alpha=0.1):
    """Run coordinate-wise conformal prediction"""
    print("\n" + "="*70)
    print("METHOD 1: Coordinate-wise Conformal Prediction")
    print("="*70)
    
    start_time = time.time()
    
    # Get predictions
    Y_pred_calib = astra_wrapper.predict(X_train)
    Y_pred_test = astra_wrapper.predict(X_test)
    
    # Initialize and run coordinate-wise CP
    cwcp = CoordinateWiseCP(alpha=alpha, use_quantile_regression=True)
    cwcp.calibrate(Y_train, Y_pred_calib)
    cwcp.predict_intervals(Y_pred_test, Y_test)
    coverage, volume = cwcp.evaluate_coverage(Y_test)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Avg Volume: {volume:.2e}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'method': 'coordinate_wise',
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time
    }


def run_multidimspci(astra_wrapper, X_train, Y_train, X_test, Y_test, alpha=0.1):
    """Run MultiDimSPCI"""
    print("\n" + "="*70)
    print("METHOD 2: MultiDimSPCI")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize SPCI
    X_train_flat = X_train['past_trajectories'].reshape(len(X_train['past_trajectories']), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=astra_wrapper
    )
    
    spci.r = None  # Full rank
    spci.use_local_ellipsoid = False
    
    # Get predictions and compute residuals
    Y_pred_calib = astra_wrapper.predict(X_train)
    Y_pred_test = astra_wrapper.predict(X_test)
    
    residuals_calib = Y_train - Y_pred_calib
    residuals_test = Y_test - Y_pred_test
    
    n_calib = len(residuals_calib)
    n_test = len(residuals_test)
    
    spci.Ensemble_train_interval_centers = Y_pred_calib
    spci.Ensemble_pred_interval_centers = Y_pred_test
    spci.Ensemble_online_resid[:n_calib] = residuals_calib
    spci.Ensemble_online_resid[n_calib:n_calib+n_test] = residuals_test
    
    # Compute nonconformity scores
    spci.get_test_et = False
    spci.train_et = spci.get_et(residuals_calib)
    spci.get_test_et = True
    spci.test_et = spci.get_et(residuals_test)
    spci.all_et = np.concatenate([spci.train_et, spci.test_et])
    
    # Compute prediction intervals
    spci.compute_Widths_Ensemble_online(
        alpha=alpha,
        stride=1,
        smallT=False,
        past_window=10,
        use_SPCI=True
    )
    
    coverage, volume = spci.get_results()
    elapsed_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Avg Volume: {volume:.2e}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'method': 'multidimspci',
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plot(results_cw, results_spci, alpha, subset):
    """Create comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ['Coordinate-wise', 'MultiDimSPCI']
    coverages = [results_cw['coverage'], results_spci['coverage']]
    volumes = [results_cw['volume'], results_spci['volume']]
    times = [results_cw['time'], results_spci['time']]
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Coverage comparison
    ax = axes[0]
    bars = ax.bar(methods, [c*100 for c in coverages], color=colors, alpha=0.8, edgecolor='black')
    ax.axhline((1-alpha)*100, color='red', linestyle='--', linewidth=2, label='Target')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title(f'Coverage Comparison - {subset.upper()}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Volume comparison
    ax = axes[1]
    bars = ax.bar(methods, volumes, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Avg Volume', fontsize=12)
    ax.set_title('Prediction Region Size', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Time comparison
    ax = axes[2]
    bars = ax.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("BASELINE COMPARISON - HOTEL SUBSET")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Subset: {SUBSET}")
    print(f"  Alpha: {ALPHA} (Target coverage: {(1-ALPHA)*100:.0f}%)")
    print(f"  Device: {DEVICE}")
    print(f"  Split: {SPLIT_RATIO*100:.0f}% calibration, {(1-SPLIT_RATIO)*100:.0f}% test")
    
    # Load config
    print(f"\nLoading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    cfg.device = DEVICE
    cfg.device_list = [DEVICE] if DEVICE != 'cpu' else []
    
    # Set random seed
    set_seed(42)
    
    # Prepare data
    X_train, Y_train, X_test, Y_test = prepare_data(cfg, SUBSET, SPLIT_RATIO)
    
    # Initialize ASTRA
    print("\nInitializing ASTRA...")
    astra_wrapper = ASTRASklearnWrapper(
        config_path=CONFIG_PATH,
        pretrained_weights_path=f'pretrained_astra_weights/{SUBSET}_best_model.pth',
        unet_weights_path='pretrained_unet_weights/eth_unet_model_best.pt',
        use_pretrained_unet=True,
        device=DEVICE,
        dataset='ETH_UCY'
    )
    astra_wrapper.fit(X_train, Y_train)
    print("✓ ASTRA initialized")
    
    # Run both methods
    try:
        results_cw = run_coordinatewise(astra_wrapper, X_train, Y_train, X_test, Y_test, ALPHA)
    except Exception as e:
        print(f"\n✗ Coordinate-wise failed: {str(e)}")
        return
    
    try:
        results_spci = run_multidimspci(astra_wrapper, X_train, Y_train, X_test, Y_test, ALPHA)
    except Exception as e:
        print(f"\n✗ MultiDimSPCI failed: {str(e)}")
        return
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\nTarget Coverage: {(1-ALPHA)*100:.1f}%\n")
    
    print("Coordinate-wise CP:")
    print(f"  Coverage: {results_cw['coverage']*100:.2f}%")
    print(f"  Volume: {results_cw['volume']:.2e}")
    print(f"  Time: {results_cw['time']:.2f}s")
    
    print("\nMultiDimSPCI:")
    print(f"  Coverage: {results_spci['coverage']*100:.2f}%")
    print(f"  Volume: {results_spci['volume']:.2e}")
    print(f"  Time: {results_spci['time']:.2f}s")
    
    # Calculate improvements
    if results_cw['volume'] > 0:
        vol_reduction = (1 - results_spci['volume'] / results_cw['volume']) * 100
        print(f"\nImprovement:")
        print(f"  Volume reduction: {vol_reduction:.4f}%")
        
        if vol_reduction > 99:
            factor = results_cw['volume'] / results_spci['volume']
            print(f"  (MultiDimSPCI is ~{factor:.2e}x more efficient)")
    
    if results_cw['time'] > 0:
        speedup = results_cw['time'] / results_spci['time']
        print(f"  Speedup: {speedup:.1f}x faster")
    
    # Save results
    os.makedirs('results/baseline_hotel', exist_ok=True)
    
    # CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame([
        {
            'subset': SUBSET,
            'method': 'coordinate_wise',
            'coverage': results_cw['coverage'],
            'volume': results_cw['volume'],
            'time': results_cw['time'],
            'alpha': ALPHA,
            'n_calib': len(Y_train),
            'n_test': len(Y_test)
        },
        {
            'subset': SUBSET,
            'method': 'multidimspci',
            'coverage': results_spci['coverage'],
            'volume': results_spci['volume'],
            'time': results_spci['time'],
            'alpha': ALPHA,
            'n_calib': len(Y_train),
            'n_test': len(Y_test)
        }
    ])
    
    csv_path = f'results/baseline_hotel/baseline_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Plot
    fig = create_comparison_plot(results_cw, results_spci, ALPHA, SUBSET)
    fig_path = f'results/baseline_hotel/baseline_comparison_{timestamp}.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved to: {fig_path}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
