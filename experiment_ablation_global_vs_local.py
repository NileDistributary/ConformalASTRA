"""
Ablation Study 5: Global vs Local Ellipsoids
Tests whether local ellipsoid adaptation improves performance.
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
from experiment_utils import (
    save_results_to_csv, 
    save_figure,
    print_experiment_header
)


def prepare_data(cfg, subset='eth', split_ratio=0.8):
    """Load and prepare data"""
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
    
    return X_train, Y_train, X_test, Y_test


def run_with_ellipsoid_type(astra_wrapper, X_train, Y_train, X_test, Y_test,
                            use_local, alpha=0.1, rank=12, past_window=100):
    """Run MultiDimSPCI with global or local ellipsoids"""
    ellipsoid_type = "Local Ellipsoids" if use_local else "Global Ellipsoid"
    
    print(f"\n{'='*70}")
    print(f"Testing: {ellipsoid_type}")
    print('='*70)
    
    start_time = time.time()
    
    X_train_flat = X_train['past_trajectories'].reshape(len(X_train['past_trajectories']), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=astra_wrapper
    )
    
    spci.r = rank
    # KEY: Set use_local_ellipsoid parameter
    spci.use_local_ellipsoid = use_local  # This is what we're testing!
    
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
    
    spci.get_test_et = False
    spci.train_et = spci.get_et(residuals_calib)
    spci.get_test_et = True
    spci.test_et = spci.get_et(residuals_test)
    spci.all_et = np.concatenate([spci.train_et, spci.test_et])
    
    spci.compute_Widths_Ensemble_online(
        alpha=alpha,
        stride=1,
        smallT=False,
        past_window=past_window,
        use_SPCI=True
    )
    
    coverage, volume = spci.get_results()
    elapsed_time = time.time() - start_time
    
    print(f"Results:")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Volume: {volume:.2e}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'ellipsoid_type': ellipsoid_type,
        'use_local': use_local,
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time
    }


def plot_ellipsoid_comparison(results):
    """Create visualization comparing global vs local ellipsoids"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    types = [r['ellipsoid_type'] for r in results]
    type_labels = ['Global\nEllipsoid', 'Local\nEllipsoids']
    coverages = [r['coverage'] * 100 for r in results]
    volumes = [r['volume'] for r in results]
    times = [r['time'] for r in results]
    
    colors = ['#9B59B6', '#3498DB']
    
    # Coverage comparison
    ax = axes[0]
    bars = ax.bar(type_labels, coverages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Coverage Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Volume comparison
    ax = axes[1]
    bars = ax.bar(type_labels, volumes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Average Volume', fontsize=12)
    ax.set_title('Volume Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, volumes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement percentage
    if len(volumes) == 2:
        improvement = (1 - volumes[1] / volumes[0]) * 100
        ax.text(0.5, 0.95, f'Volume reduction: {improvement:.1f}%',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, fontweight='bold')
    
    # Time comparison
    ax = axes[2]
    bars = ax.bar(type_labels, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print_experiment_header("ABLATION STUDY: GLOBAL VS LOCAL ELLIPSOIDS")
    
    # Configuration
    config_path = 'configs/eth.yaml'
    subset = 'eth'
    alpha = 0.1
    rank = 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(42)
    
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = munch.munchify(cfg)
    
    print("Loading data...")
    X_train, Y_train, X_test, Y_test = prepare_data(cfg, subset=subset)
    
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
    
    # Test both ellipsoid types
    all_results = []
    
    # 1. Global Ellipsoid
    result_global = run_with_ellipsoid_type(
        astra_wrapper, X_train, Y_train, X_test, Y_test,
        use_local=False, alpha=alpha, rank=rank
    )
    all_results.append(result_global)
    
    save_results_to_csv({
        'experiment': 'global_vs_local',
        'subset': subset,
        'alpha': alpha,
        'rank': rank,
        **result_global
    }, 'ablation_global_vs_local', append=True)
    
    # 2. Local Ellipsoids
    result_local = run_with_ellipsoid_type(
        astra_wrapper, X_train, Y_train, X_test, Y_test,
        use_local=True, alpha=alpha, rank=rank
    )
    all_results.append(result_local)
    
    save_results_to_csv({
        'experiment': 'global_vs_local',
        'subset': subset,
        'alpha': alpha,
        'rank': rank,
        **result_local
    }, 'ablation_global_vs_local', append=True)
    
    # Create and save visualization
    fig = plot_ellipsoid_comparison(all_results)
    save_figure(fig, 'ablation_global_vs_local', 'ellipsoid_comparison', timestamp=True)
    # Print summary
    print("\n" + "="*70)
    print("GLOBAL VS LOCAL ELLIPSOIDS SUMMARY")
    print("="*70)
    for result in all_results:
        print(f"\n{result['ellipsoid_type']}:")
        print(f"  Coverage: {result['coverage']*100:.2f}%")
        print(f"  Volume: {result['volume']:.2e}")
        print(f"  Time: {result['time']:.2f}s")
    
    # Calculate improvements
    if len(all_results) == 2:
        vol_improvement = (1 - result_local['volume'] / result_global['volume']) * 100
        time_overhead = ((result_local['time'] / result_global['time']) - 1) * 100
        print(f"\nLocal Ellipsoid Improvements:")
        print(f"  Volume reduction: {vol_improvement:.1f}%")
        print(f"  Coverage change: {(result_local['coverage'] - result_global['coverage'])*100:.2f} percentage points")
        print(f"  Time overhead: {time_overhead:.1f}%")
    print("="*70)