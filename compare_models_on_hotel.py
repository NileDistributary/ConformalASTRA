"""
Cross-Model Comparison: ETH Model vs Hotel Model on Hotel Data

Tests whether the hotel-specific model (LOSO trained) performs better than
a general ETH model when both are evaluated on hotel test data.

Evaluates:
1. Standard ASTRA metrics (ADE, FDE)
2. Conformal prediction (coverage, volume)

Usage:
    set CUDA_VISIBLE_DEVICES=-1
    python compare_models_on_hotel.py
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
from icecream import ic
ic.disable() #Disable Icecream verbose outputs to reduce crowding of test terminal


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_PATH = 'configs/hotel.yaml'
TEST_SUBSET = 'hotel'  # Testing on hotel data
ALPHA = 0.1
SPLIT_RATIO = 0.90
DEVICE = 'cpu'

# Model configurations to test
MODELS_TO_TEST = {
    'hotel_model': {
        'name': 'Hotel Model (LOSO)',
        'weights': 'pretrained_astra_weights/hotel_best_model.pth',
        'description': 'Trained WITHOUT hotel data'
    },
    'eth_model': {
        'name': 'ETH Model (General)',
        'weights': 'pretrained_astra_weights/eth_best_model.pth',
        'description': 'Trained WITHOUT eth data'
    }
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(cfg, subset, split_ratio=0.90):
    """Load and prepare data"""
    print(f"\nLoading {subset} test data...")
    
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
    
    return X_train, Y_train, X_test, Y_test, full_dataset


# ============================================================================
# STANDARD ASTRA EVALUATION
# ============================================================================

def compute_ade_fde(predictions, ground_truth):
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE)
    
    Args:
        predictions: (N, pred_len*2) flattened predictions
        ground_truth: (N, pred_len*2) flattened ground truth
    """
    # Reshape to (N, pred_len, 2)
    pred_len = predictions.shape[1] // 2
    preds = predictions.reshape(-1, pred_len, 2)
    gt = ground_truth.reshape(-1, pred_len, 2)
    
    # Compute displacement errors at each timestep
    displacements = np.sqrt(np.sum((preds - gt)**2, axis=2))  # (N, pred_len)
    
    # ADE: Average across all timesteps and samples
    ade = np.mean(displacements)
    
    # FDE: Final displacement (last timestep) averaged across samples
    fde = np.mean(displacements[:, -1])
    
    return ade, fde


def evaluate_standard_metrics(astra_wrapper, X_test, Y_test):
    """Evaluate standard ASTRA metrics"""
    print("\n  Computing ADE/FDE...")
    
    # Get predictions
    Y_pred = astra_wrapper.predict(X_test)
    
    # Compute metrics
    ade, fde = compute_ade_fde(Y_pred, Y_test)
    
    print(f"    ADE: {ade:.4f}")
    print(f"    FDE: {fde:.4f}")
    
    return {
        'ade': ade,
        'fde': fde
    }


# ============================================================================
# CONFORMAL PREDICTION EVALUATION
# ============================================================================

def evaluate_conformal_prediction(astra_wrapper, X_train, Y_train, X_test, Y_test, alpha=0.1):
    """Evaluate MultiDimSPCI conformal prediction"""
    print("\n  Computing conformal prediction metrics...")
    
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
    
    print(f"    Coverage: {coverage*100:.2f}%")
    print(f"    Volume: {volume:.2e}")
    print(f"    Time: {elapsed_time:.2f}s")
    
    return {
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time
    }


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model_key, model_info, cfg, X_train, Y_train, X_test, Y_test, alpha):
    """Evaluate a single model on all metrics"""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_info['name']}")
    print(f"{'='*70}")
    print(f"Weights: {model_info['weights']}")
    print(f"Description: {model_info['description']}")
    
    # Check if weights exist
    if not os.path.exists(model_info['weights']):
        print(f"\n✗ Model weights not found: {model_info['weights']}")
        return None
    
    # Initialize ASTRA with this model
    print("\nInitializing ASTRA...")
    try:
        astra_wrapper = ASTRASklearnWrapper(
            config_path=CONFIG_PATH,
            pretrained_weights_path=model_info['weights'],
            unet_weights_path='pretrained_unet_weights/eth_unet_model_best.pt',
            use_pretrained_unet=True,
            device=DEVICE,
            dataset='ETH_UCY'
        )
        astra_wrapper.fit(X_train, Y_train)
        print("✓ ASTRA initialized")
    except Exception as e:
        print(f"✗ Failed to initialize ASTRA: {str(e)}")
        return None
    
    # Evaluate standard metrics
    print("\n" + "-"*70)
    print("STANDARD METRICS")
    print("-"*70)
    try:
        standard_results = evaluate_standard_metrics(astra_wrapper, X_test, Y_test)
    except Exception as e:
        print(f"✗ Standard metrics failed: {str(e)}")
        standard_results = {'ade': None, 'fde': None}
    
    # Evaluate conformal prediction
    print("\n" + "-"*70)
    print("CONFORMAL PREDICTION")
    print("-"*70)
    try:
        conformal_results = evaluate_conformal_prediction(
            astra_wrapper, X_train, Y_train, X_test, Y_test, alpha
        )
    except Exception as e:
        print(f"✗ Conformal prediction failed: {str(e)}")
        conformal_results = {'coverage': None, 'volume': None, 'time': None}
    
    # Combine results
    return {
        'model': model_key,
        'model_name': model_info['name'],
        'weights': model_info['weights'],
        **standard_results,
        **conformal_results
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plots(results_list, test_subset, alpha):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    models = [r['model_name'] for r in results_list]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # Plot 1: ADE Comparison
    ax = axes[0]
    ades = [r['ade'] for r in results_list]
    bars = ax.bar(models, ades, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax.set_ylabel('ADE', fontsize=12)
    ax.set_title(f'Average Displacement Error - {test_subset.upper()} Data', 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, ades):
        if val is not None:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: FDE Comparison
    ax = axes[1]
    fdes = [r['fde'] for r in results_list]
    bars = ax.bar(models, fdes, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax.set_ylabel('FDE', fontsize=12)
    ax.set_title(f'Final Displacement Error - {test_subset.upper()} Data', 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, fdes):
        if val is not None:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Coverage Comparison
    ax = axes[2]
    coverages = [r['coverage']*100 if r['coverage'] is not None else 0 for r in results_list]
    bars = ax.bar(models, coverages, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    target_coverage = (1-alpha)*100
    ax.axhline(target_coverage, color='red', linestyle='--', linewidth=2, 
               label=f'Target ({target_coverage:.0f}%)')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Conformal Prediction Coverage', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, coverages):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Volume Comparison
    ax = axes[3]
    volumes = [r['volume'] if r['volume'] is not None else 0 for r in results_list]
    bars = ax.bar(models, volumes, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Volume', fontsize=12)
    ax.set_title('Prediction Region Volume', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CROSS-MODEL COMPARISON: ETH vs HOTEL Models on HOTEL Data")
    print("="*70)
    print(f"\nTest Dataset: {TEST_SUBSET}")
    print(f"Alpha: {ALPHA} (Target coverage: {(1-ALPHA)*100:.0f}%)")
    print(f"Device: {DEVICE}")
    
    print(f"\nModels to test:")
    for key, info in MODELS_TO_TEST.items():
        print(f"  - {info['name']}: {info['weights']}")
    
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    cfg.device = DEVICE
    cfg.device_list = [DEVICE] if DEVICE != 'cpu' else []
    
    set_seed(42)
    
    # Prepare data (hotel test data)
    X_train, Y_train, X_test, Y_test, dataset = prepare_data(cfg, TEST_SUBSET, SPLIT_RATIO)
    
    # Evaluate each model
    all_results = []
    for model_key, model_info in MODELS_TO_TEST.items():
        result = evaluate_model(
            model_key, model_info, cfg,
            X_train, Y_train, X_test, Y_test, ALPHA
        )
        if result is not None:
            all_results.append(result)
    
    # Print comparison summary
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\nTest Data: {TEST_SUBSET} ({len(Y_test)} test samples)")
    print(f"Target Coverage: {(1-ALPHA)*100:.0f}%\n")
    
    # Create comparison table
    print(f"{'Model':<25} {'ADE':<10} {'FDE':<10} {'Coverage':<12} {'Volume':<15}")
    print("-"*70)
    
    for result in all_results:
        ade_str = f"{result['ade']:.4f}" if result['ade'] is not None else "N/A"
        fde_str = f"{result['fde']:.4f}" if result['fde'] is not None else "N/A"
        cov_str = f"{result['coverage']*100:.2f}%" if result['coverage'] is not None else "N/A"
        vol_str = f"{result['volume']:.2e}" if result['volume'] is not None else "N/A"
        
        print(f"{result['model_name']:<25} {ade_str:<10} {fde_str:<10} {cov_str:<12} {vol_str:<15}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if len(all_results) >= 2:
        hotel_result = next((r for r in all_results if 'hotel' in r['model'].lower()), None)
        eth_result = next((r for r in all_results if 'eth' in r['model'].lower()), None)
        
        if hotel_result and eth_result and hotel_result['ade'] is not None and eth_result['ade'] is not None:
            print("\nStandard Metrics (on hotel data):")
            
            ade_diff = ((eth_result['ade'] - hotel_result['ade']) / hotel_result['ade']) * 100
            fde_diff = ((eth_result['fde'] - hotel_result['fde']) / hotel_result['fde']) * 100
            
            print(f"  ADE: Hotel={hotel_result['ade']:.4f}, ETH={eth_result['ade']:.4f}")
            if ade_diff > 0:
                print(f"       → Hotel model is {ade_diff:.1f}% better")
            else:
                print(f"       → ETH model is {-ade_diff:.1f}% better")
            
            print(f"  FDE: Hotel={hotel_result['fde']:.4f}, ETH={eth_result['fde']:.4f}")
            if fde_diff > 0:
                print(f"       → Hotel model is {fde_diff:.1f}% better")
            else:
                print(f"       → ETH model is {-fde_diff:.1f}% better")
        
        if hotel_result and eth_result and hotel_result['coverage'] is not None and eth_result['coverage'] is not None:
            print("\nConformal Prediction (on hotel data):")
            print(f"  Coverage: Hotel={hotel_result['coverage']*100:.2f}%, ETH={eth_result['coverage']*100:.2f}%")
            print(f"  Volume: Hotel={hotel_result['volume']:.2e}, ETH={eth_result['volume']:.2e}")
            
            if hotel_result['volume'] < eth_result['volume']:
                vol_improvement = ((eth_result['volume'] - hotel_result['volume']) / eth_result['volume']) * 100
                print(f"       → Hotel model has {vol_improvement:.1f}% smaller uncertainty regions")
            else:
                vol_improvement = ((hotel_result['volume'] - eth_result['volume']) / hotel_result['volume']) * 100
                print(f"       → ETH model has {vol_improvement:.1f}% smaller uncertainty regions")
    
    print("\nConclusion:")
    print("  This comparison tests whether LOSO (holding out specific scenes)")
    print("  improves performance, or if a general model works just as well.")
    
    # Save results
    os.makedirs('results/cross_model_comparison', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = f'results/cross_model_comparison/comparison_{TEST_SUBSET}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Save plot
    if len(all_results) > 0:
        fig = create_comparison_plots(all_results, TEST_SUBSET, ALPHA)
        fig_path = f'results/cross_model_comparison/comparison_{TEST_SUBSET}_{timestamp}.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved to: {fig_path}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
