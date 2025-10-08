"""
Ablation Study 2: Calibration Set Size
Tests how the size of calibration data affects coverage and volume.
"""
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from utils.misc import set_seed
from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI
from astra_wrapper import ASTRASklearnWrapper
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_utils import (
    save_results_to_csv, 
    save_figure,
    print_experiment_header
)


def prepare_data_with_split(cfg, subset='eth', calib_ratio=0.5, test_ratio=0.3):
    """
    Load data with flexible calibration/test split
    
    Args:
        calib_ratio: Fraction of data for calibration
        test_ratio: Fraction of data for testing (remaining is unused)
    """
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
    n_calib = int(n_samples * calib_ratio)
    n_test = int(n_samples * test_ratio)
    
    # Ensure we have enough data
    if n_calib + n_test > n_samples:
        n_test = n_samples - n_calib
    
    X_calib = {
        'past_trajectories': all_past_trajectories[:n_calib],
        'images': all_images[:n_calib],
        'num_valid': all_num_valid[:n_calib]
    }
    Y_calib = all_future_trajectories[:n_calib].reshape(n_calib, -1)
    
    X_test = {
        'past_trajectories': all_past_trajectories[n_calib:n_calib+n_test],
        'images': all_images[n_calib:n_calib+n_test],
        'num_valid': all_num_valid[n_calib:n_calib+n_test]
    }
    Y_test = all_future_trajectories[n_calib:n_calib+n_test].reshape(n_test, -1)
    
    return X_calib, Y_calib, X_test, Y_test


def run_with_calib_size(astra_wrapper, X_train, Y_train, X_test, Y_test,
                        calib_size_fraction, alpha=0.1, rank=None):
    """Run MultiDimSPCI with specific calibration set size"""
    # Subsample calibration data
    n_calib_total = len(Y_train)
    n_calib_use = max(50, int(n_calib_total * calib_size_fraction))  # Minimum 50 samples
    
    if n_calib_use > n_calib_total:
        n_calib_use = n_calib_total
    
    print(f"\n{'='*70}")
    print(f"Calibration Size: {n_calib_use}/{n_calib_total} "
          f"({calib_size_fraction*100:.1f}%)")
    print('='*70)
    
    # Subsample calibration data
    indices = np.random.choice(n_calib_total, n_calib_use, replace=False)
    X_train_sub = {k: v[indices] for k, v in X_train.items()}
    Y_train_sub = Y_train[indices]
    
    start_time = time.time()
    
    X_train_flat = X_train_sub['past_trajectories'].reshape(len(X_train_sub['past_trajectories']), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train_sub,
        Y_predict=Y_test,
        fit_func=astra_wrapper
    )
    
    spci.r = rank
    spci.use_local_ellipsoid = False
    
    Y_pred_calib = astra_wrapper.predict(X_train_sub)
    Y_pred_test = astra_wrapper.predict(X_test)
    
    residuals_calib = Y_train_sub - Y_pred_calib
    residuals_test = Y_test - Y_pred_test
    
    n_test = len(residuals_test)
    
    spci.Ensemble_train_interval_centers = Y_pred_calib
    spci.Ensemble_pred_interval_centers = Y_pred_test
    spci.Ensemble_online_resid[:n_calib_use] = residuals_calib
    spci.Ensemble_online_resid[n_calib_use:n_calib_use+n_test] = residuals_test
    
    spci.get_test_et = False
    spci.train_et = spci.get_et(residuals_calib)
    spci.get_test_et = True
    spci.test_et = spci.get_et(residuals_test)
    spci.all_et = np.concatenate([spci.train_et, spci.test_et])
    
    spci.compute_Widths_Ensemble_online(
        alpha=alpha,
        stride=1,
        smallT=False,
        past_window=10,
        use_SPCI=True
    )
    
    coverage, volume = spci.get_results()
    elapsed_time = time.time() - start_time
    
    print(f"Results:")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Volume: {volume:.2e}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'calib_size': n_calib_use,
        'calib_fraction': calib_size_fraction,
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time
    }


def plot_calib_size_results(results):
    """Create visualization for calibration size ablation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    calib_sizes = [r['calib_size'] for r in results]
    coverages = [r['coverage'] * 100 for r in results]
    volumes = [r['volume'] for r in results]
    times = [r['time'] for r in results]
    
    # Coverage vs Calibration Size
    ax = axes[0]
    ax.plot(calib_sizes, coverages, 'o-', linewidth=2, markersize=8)
    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_xlabel('Calibration Set Size', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Coverage vs Calibration Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Volume vs Calibration Size
    ax = axes[1]
    ax.plot(calib_sizes, volumes, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Calibration Set Size', fontsize=12)
    ax.set_ylabel('Average Volume', fontsize=12)
    ax.set_title('Volume vs Calibration Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Time vs Calibration Size
    ax = axes[2]
    ax.plot(calib_sizes, times, '^-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Calibration Set Size', fontsize=12)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Time vs Calibration Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print_experiment_header("ABLATION STUDY: CALIBRATION SET SIZE")
    
    # Configuration
    config_path = 'configs/eth.yaml'
    subset = 'eth'
    alpha = 0.1
    rank = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different calibration sizes
    calib_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    set_seed(42)
    
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = munch.munchify(cfg)
    
    print("Loading data...")
    # Use large calibration set initially, will subsample
    X_train, Y_train, X_test, Y_test = prepare_data_with_split(
        cfg, subset=subset, calib_ratio=0.9, test_ratio=0.1
    )
    
    print("Initializing ASTRA...")
    astra_wrapper = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=f'./pretrained_astra_weights/{subset}_best_model.pth',
        unet_weights_path='./pretrained_unet_weights/eth_unet_model_best.pt',
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    astra_wrapper.fit(X_train, Y_train)
    
    # Run experiments for each calibration size
    all_results = []
    for frac in calib_fractions:
        result = run_with_calib_size(
            astra_wrapper, X_train, Y_train, X_test, Y_test,
            calib_size_fraction=frac, alpha=alpha, rank=rank
        )
        all_results.append(result)
        
        # Save individual result
        save_results_to_csv({
            'experiment': 'calibration_size_ablation',
            'subset': subset,
            'alpha': alpha,
            'rank': rank,
            **result
        }, 'ablation_calibration_size', append=True)
    
    # Create and save visualization
    fig = plot_calib_size_results(all_results)
    save_figure(fig, 'ablation_calibration_size', 'calib_size_comparison', timestamp=True)
    # Print summary
    print("\n" + "="*70)
    print("CALIBRATION SIZE ABLATION SUMMARY")
    print("="*70)
    for result in all_results:
        print(f"Calib Size {result['calib_size']} ({result['calib_fraction']*100:.1f}%): "
              f"Coverage={result['coverage']*100:.2f}%, "
              f"Volume={result['volume']:.2e}, "
              f"Time={result['time']:.2f}s")
    print("="*70)