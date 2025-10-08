"""
Experiment 1: Baseline Comparison
Compare coordinate-wise conformal prediction vs MultiDimSPCI

This script runs both methods and saves:
- CSV with coverage and volume metrics
- Comparison visualization figures
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
import matplotlib.pyplot as plt
from utils.misc import set_seed
from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI
from astra_wrapper import ASTRASklearnWrapper
import time
from icecream import ic
ic.disable()

# Import our utilities
from experiment_utils import (
    save_results_to_csv, 
    save_figure, 
    save_config,
    print_experiment_header,
    print_experiment_summary
)
from baseline_coordinatewise import CoordinateWiseCP


def prepare_data(cfg, subset='eth', split_ratio=0.8):
    """Load and prepare data (same as castra3.py)"""
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
    
    loader = data.DataLoader(
        full_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    # Collect all data
    all_past_trajectories = []
    all_future_trajectories = []
    all_images = []
    all_num_valid = []
    
    for batch_idx, batch in enumerate(loader):
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


def run_coordinatewise(astra_wrapper, X_train, Y_train, X_test, Y_test, 
                       alpha=0.1, use_qr=True):
    """Run coordinate-wise conformal prediction"""
    print("\n" + "="*70)
    print("RUNNING: Coordinate-wise Conformal Prediction")
    print("="*70)
    
    start_time = time.time()
    
    # Get predictions
    Y_pred_calib = astra_wrapper.predict(X_train)
    Y_pred_test = astra_wrapper.predict(X_test)
    
    # Initialize and run coordinate-wise CP
    cwcp = CoordinateWiseCP(alpha=alpha, use_quantile_regression=use_qr)
    cwcp.calibrate(Y_train, Y_pred_calib)
    cwcp.predict_intervals(Y_pred_test, Y_test)
    coverage, volume = cwcp.evaluate_coverage(Y_test)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nCoordinate-wise Results:")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Avg Volume: {volume:.2e}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time,
        'predictor': cwcp
    }


def run_multidimspci(astra_wrapper, X_train, Y_train, X_test, Y_test,
                     alpha=0.1, rank=12, use_local=False):
    """Run MultiDimSPCI (from castra3.py)"""
    print("\n" + "="*70)
    print("RUNNING: MultiDimSPCI")
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
    
    # Set rank
    spci.r = rank
    spci.use_local_ellipsoid = use_local
    
    # Compute residuals with fixed predictor
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
        past_window=100,
        use_SPCI=True
    )
    
    coverage, volume = spci.get_results()
    elapsed_time = time.time() - start_time
    
    print(f"\nMultiDimSPCI Results:")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Avg Volume: {volume:.2e}")
    print(f"  Time: {elapsed_time:.2f}s")
    
    return {
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time,
        'predictor': spci
    }


def create_comparison_plot(results_cw, results_spci, alpha):
    """Create comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ['Coordinate-wise', 'MultiDimSPCI']
    coverages = [results_cw['coverage'], results_spci['coverage']]
    volumes = [results_cw['volume'], results_spci['volume']]
    times = [results_cw['time'], results_spci['time']]
    
    # Coverage comparison
    ax = axes[0]
    bars = ax.bar(methods, [c*100 for c in coverages], color=['#FF6B6B', '#4ECDC4'])
    ax.axhline((1-alpha)*100, color='red', linestyle='--', linewidth=2, label='Target')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Coverage Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Volume comparison
    ax = axes[1]
    bars = ax.bar(methods, volumes, color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('Avg Volume', fontsize=12)
    ax.set_title('Prediction Region Size', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Add efficiency improvement
    improvement = (1 - volumes[1]/volumes[0]) * 100
    ax.text(0.5, 0.95, f'MultiDimSPCI: {improvement:.1f}% smaller',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=10, fontweight='bold')
    
    # Time comparison
    ax = axes[2]
    bars = ax.bar(methods, times, color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print_experiment_header("Baseline Comparison: Coordinate-wise vs MultiDimSPCI")
    
    # Configuration
    config_path = './configs/eth.yaml'
    subset = 'eth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alpha = 0.1
    rank = 12
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    cfg.SUBSET = subset
    cfg.device = device
    cfg.device_list = [device] if device != 'cpu' else []
    
    set_seed(cfg.TRAIN.SEED)
    
    # Save experiment config
    exp_config = {
        'experiment': 'baseline_comparison',
        'subset': subset,
        'alpha': alpha,
        'rank': rank,
        'device': device,
        'split_ratio': 0.8
    }
    save_config(exp_config, 'baseline_comparison')
    
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
    
    # Run both methods
    results_cw = run_coordinatewise(astra_wrapper, X_train, Y_train, X_test, Y_test, alpha=alpha)
    results_spci = run_multidimspci(astra_wrapper, X_train, Y_train, X_test, Y_test, 
                                     alpha=alpha, rank=rank)
    
    # Save results to CSV
    combined_results = {
        'method': 'coordinate_wise',
        'subset': subset,
        'alpha': alpha,
        'coverage': results_cw['coverage'],
        'volume': results_cw['volume'],
        'time': results_cw['time'],
        'n_calib': len(Y_train),
        'n_test': len(Y_test)
    }
    save_results_to_csv(combined_results, 'baseline_comparison', append=True)
    
    combined_results['method'] = 'multidimspci'
    combined_results['rank'] = rank
    combined_results['coverage'] = results_spci['coverage']
    combined_results['volume'] = results_spci['volume']
    combined_results['time'] = results_spci['time']
    save_results_to_csv(combined_results, 'baseline_comparison', append=True)
    
    # Create and save comparison plot
    fig = create_comparison_plot(results_cw, results_spci, alpha)
    save_figure(fig, 'baseline_comparison', 'coverage_volume_time', timestamp=True)
    plt.show()
    
    # Print final summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Target Coverage: {(1-alpha)*100:.1f}%")
    print(f"\nCoordinate-wise:")
    print(f"  Coverage: {results_cw['coverage']*100:.2f}%")
    print(f"  Volume: {results_cw['volume']:.2e}")
    print(f"\nMultiDimSPCI:")
    print(f"  Coverage: {results_spci['coverage']*100:.2f}%")
    print(f"  Volume: {results_spci['volume']:.2e}")
    print(f"\nImprovement:")
    improvement = (1 - results_spci['volume']/results_cw['volume']) * 100
    print(f"  Volume reduction: {improvement:.1f}%")
    print("="*70)