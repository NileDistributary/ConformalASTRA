"""
Ablation Study 1: Rank Approximation
Tests how covariance matrix rank (r) affects coverage and volume.
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


def run_with_rank(astra_wrapper, X_train, Y_train, X_test, Y_test, 
                  rank, alpha=0.1, past_window=100):
    """Run MultiDimSPCI with specific rank"""
    print(f"\n{'='*70}")
    print(f"Testing Rank: {rank if rank is not None else 'Full'}")
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
    
    # Set rank
    spci.r = rank
    spci.use_local_ellipsoid = False
    
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
        'rank': rank if rank is not None else 'full',
        'coverage': coverage,
        'volume': volume,
        'time': elapsed_time
    }


def plot_rank_results(results):
    """Create visualization for rank ablation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ranks = [r['rank'] for r in results]
    rank_labels = [str(r) if r != 'full' else 'Full' for r in ranks]
    coverages = [r['coverage'] * 100 for r in results]
    volumes = [r['volume'] for r in results]
    times = [r['time'] for r in results]
    
    # Coverage vs Rank
    ax = axes[0]
    ax.plot(range(len(ranks)), coverages, 'o-', linewidth=2, markersize=8)
    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_xlabel('Rank Configuration', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Coverage vs Rank', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(rank_labels)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Volume vs Rank
    ax = axes[1]
    ax.plot(range(len(ranks)), volumes, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Rank Configuration', fontsize=12)
    ax.set_ylabel('Average Volume', fontsize=12)
    ax.set_title('Volume vs Rank', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(rank_labels)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Time vs Rank
    ax = axes[2]
    ax.plot(range(len(ranks)), times, '^-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Rank Configuration', fontsize=12)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Time vs Rank', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(rank_labels)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print_experiment_header("ABLATION STUDY: RANK APPROXIMATION")
    
    # Configuration
    config_path = 'configs/eth.yaml'
    subset = 'eth'
    alpha = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different ranks (including None for full rank)
    ranks_to_test = [4, 8, 12, 16, 20, None]
    
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
    
    # Run experiments for each rank
    all_results = []
    for rank in ranks_to_test:
        result = run_with_rank(
            astra_wrapper, X_train, Y_train, X_test, Y_test,
            rank=rank, alpha=alpha
        )
        all_results.append(result)
        
        # Save individual result
        save_results_to_csv({
            'experiment': 'rank_ablation',
            'subset': subset,
            'alpha': alpha,
            **result
        }, 'ablation_rank', append=True)
    
    # Create and save visualization
    fig = plot_rank_results(all_results)
    save_figure(fig, 'ablation_rank', 'rank_comparison', timestamp=True)
    # Print summary
    print("\n" + "="*70)
    print("RANK ABLATION SUMMARY")
    print("="*70)
    for result in all_results:
        print(f"Rank {result['rank']}: "
              f"Coverage={result['coverage']*100:.2f}%, "
              f"Volume={result['volume']:.2e}, "
              f"Time={result['time']:.2f}s")
    print("="*70)