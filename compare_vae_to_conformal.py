"""
Complete Experiment: Compare ASTRA's Stochastic VAE Outputs to 
Deterministic Conformal Prediction Intervals

EXPERIMENTAL DESIGN:
====================
1. Use FIXED pretrained ASTRA (deterministic, USE_VAE=False) to compute 
   conformal prediction intervals via MultiDimSPCI (following Algorithm 1)
   
2. Separately generate stochastic samples from ASTRA with VAE enabled
   (USE_VAE=True, generates K samples per trajectory)
   
3. Evaluate: What percentage of VAE samples fall within the conformal 
   prediction ellipsoids?

NO BOOTSTRAP - Uses single fixed model with quantile regression.
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
from utils.misc import set_seed
from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from icecream import ic
import pandas as pd
import os
ic.disable()   # Disable debug prints

from astra_wrapper import ASTRASklearnWrapper


# ============================================================================
# DATA LOADING
# ============================================================================

def prepare_real_data(cfg, subset='eth', split_ratio=0.90):
    """Load and prepare real ETH-UCY data for MultiDimSPCI."""
    # Setup transforms
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
    
    # CORRECT: mode='testing' loads ASTRA's holdout
    cfg.SUBSET = subset
    full_dataset = ETH_dataset(cfg, mode='testing', img_transforms=transforms)
    
    # Create data loader
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
    
    print("Loading data from ETH dataset...")
    for batch_idx, batch in enumerate(loader):
        past_loc, fut_loc, num_valid, imgs, gt_maps, traj_coords = batch
        
        all_past_trajectories.append(past_loc.numpy())
        all_future_trajectories.append(fut_loc.numpy())
        all_images.append(imgs.numpy())
        all_num_valid.append(num_valid.numpy())
        
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx}/{len(loader)} batches")
    
    # Concatenate all data
    all_past_trajectories = np.concatenate(all_past_trajectories, axis=0)
    all_future_trajectories = np.concatenate(all_future_trajectories, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_num_valid = np.concatenate(all_num_valid, axis=0)
    
    # Split into train/test
    n_samples = len(all_past_trajectories)
    n_train = int(n_samples * split_ratio)
    
    # Training data (calibration)
    X_train = {
        'past_trajectories': all_past_trajectories[:n_train],
        'images': all_images[:n_train],
        'num_valid': all_num_valid[:n_train]
    }
    Y_train = all_future_trajectories[:n_train]
    
    # Test data
    X_test = {
        'past_trajectories': all_past_trajectories[n_train:],
        'images': all_images[n_train:],
        'num_valid': all_num_valid[n_train:]
    }
    Y_test = all_future_trajectories[n_train:]
    
    # Flatten Y for MultiDimSPCI
    Y_train_flat = Y_train.reshape(len(Y_train), -1)
    Y_test_flat = Y_test.reshape(len(Y_test), -1)
    
    print(f"\nData shapes:")
    print(f"X_train past_trajectories: {X_train['past_trajectories'].shape}")
    print(f"Y_train: {Y_train_flat.shape}")
    print(f"X_test past_trajectories: {X_test['past_trajectories'].shape}")
    print(f"Y_test: {Y_test_flat.shape}")
    
    return X_train, Y_train_flat, X_test, Y_test_flat


# ============================================================================
# STAGE 1: COMPUTE CONFORMAL PREDICTION INTERVALS
# ============================================================================

def compute_conformal_regions(X_train, Y_train, X_test, Y_test, 
                              astra_wrapper, alpha=0.1):
    """
    Compute conformal prediction intervals using FIXED pretrained ASTRA.
    This follows Algorithm 1 from MultiDimSPCI paper (no bootstrap).
    """
    print("\n" + "="*70)
    print("STAGE 1: COMPUTING CONFORMAL PREDICTION REGIONS")
    print("="*70)
    print("Method: MultiDimSPCI with FIXED pretrained ASTRA")
    print("  - Single deterministic model (no bootstrap)")
    print("  - Quantile regression on residuals")
    print("  - Sequential prediction with rolling window")
    
    # Initialize SPCI
    X_train_flat = X_train['past_trajectories'].reshape(len(Y_train), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(Y_test), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=astra_wrapper
    )
    
    # Step 1: Get predictions from FIXED model
    print("\n[1/4] Computing predictions from fixed ASTRA model...")
    Y_pred_calib = astra_wrapper.predict(X_train)
    Y_pred_test = astra_wrapper.predict(X_test)
    
    # Step 2: Compute residuals
    print("[2/4] Computing residuals...")
    residuals_calib = Y_train - Y_pred_calib
    residuals_test = Y_test - Y_pred_test
    
    # Populate SPCI internal state
    spci.Ensemble_train_interval_centers = Y_pred_calib
    spci.Ensemble_pred_interval_centers = Y_pred_test
    n_calib = len(residuals_calib)
    n_test = len(residuals_test)
    spci.Ensemble_online_resid[:n_calib] = residuals_calib
    spci.Ensemble_online_resid[n_calib:n_calib+n_test] = residuals_test
    
    # Step 3: Compute nonconformity scores (Mahalanobis distances)
    print("[3/4] Computing covariance and nonconformity scores...")
    spci.get_test_et = False
    spci.train_et = spci.get_et(residuals_calib)
    spci.get_test_et = True
    spci.test_et = spci.get_et(residuals_test)
    spci.all_et = np.concatenate([spci.train_et, spci.test_et])
    
    print(f"  ✓ Covariance matrix computed: {spci.global_cov.shape}")
    print(f"  ✓ Calibration scores: mean={spci.train_et.mean():.4f}")
    print(f"  ✓ Test scores: mean={spci.test_et.mean():.4f}")
    
    # Step 4: Sequential prediction with quantile regression
    
    print("[4/4] Computing prediction intervals with quantile regression...")
    spci.compute_Widths_Ensemble_online(
        alpha=alpha,
        stride=1,
        smallT=False,
        past_window=10,
        use_SPCI=True
    )
    
    coverage, avg_size = spci.get_results()
    
    print("\n" + "="*70)
    print("CONFORMAL PREDICTION RESULTS")
    print("="*70)
    print(f"Target coverage: {(1-alpha)*100:.1f}%")
    print(f"Empirical coverage: {coverage*100:.2f}%")
    print(f"Average ellipsoid volume: {avg_size:.2e}")
    print("="*70)
    
    return spci


# ============================================================================
# STAGE 2: GENERATE VAE STOCHASTIC SAMPLES
# ============================================================================

def generate_vae_samples(X_test, config_path, pretrained_path, unet_path, 
                        device, subset):
    """Generate K stochastic samples from ASTRA with VAE enabled."""
    print("\n" + "="*70)
    print("STAGE 2: GENERATING VAE STOCHASTIC SAMPLES")
    print("="*70)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    
    original_vae = cfg.MODEL.USE_VAE
    cfg.MODEL.USE_VAE = True
    K = cfg.MODEL.K
    
    print(f"Loading ASTRA with VAE enabled...")
    print(f"  Original config USE_VAE: {original_vae}")
    print(f"  Overriding to USE_VAE: True")
    print(f"  K (samples per trajectory): {K}")
    
    vae_wrapper = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=pretrained_path,
        unet_weights_path=unet_path,
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    
    vae_wrapper.cfg.MODEL.USE_VAE = True
    vae_wrapper.cfg.MODEL.K = K
    
    dummy_X = {
        'past_trajectories': np.zeros((1, 8, 2)),
        'images': np.zeros((1, 224, 224, 3)),
        'num_valid': np.array([8])
    }
    vae_wrapper.fit(dummy_X, np.zeros((1, 24)))
    
    print(f"\nGenerating {K} samples for {len(X_test['past_trajectories'])} test trajectories...")
    vae_samples = vae_wrapper.predict(X_test, return_multiple_samples=True)
    
    print(f"✓ VAE samples generated: {vae_samples.shape}")
    print(f"  Expected shape: (n_test, K, 24)")
    
    return vae_samples, K


# ============================================================================
# STAGE 3: COMPUTE AGREEMENT METRICS
# ============================================================================

def compute_agreement_metrics(vae_samples, spci, alpha=0.1):
    """Compute what % of VAE samples fall within conformal ellipsoids."""
    print("\n" + "="*70)
    print("STAGE 3: COMPUTING AGREEMENT METRICS")
    print("="*70)
    
    n_samples = vae_samples.shape[0]
    K = vae_samples.shape[1]
    
    conformal_centers = spci.Ensemble_pred_interval_centers
    conformal_cov = spci.global_cov
    
    threshold_24d = chi2.ppf(1 - alpha, df=24)
    threshold_2d = chi2.ppf(1 - alpha, df=2)
    
    joint_agreements = np.zeros(n_samples)
    timestep_agreements = np.zeros((n_samples, 12))
    
    print(f"Computing agreement for {n_samples} samples...")
    
    for i in range(n_samples):
        # Joint agreement (24D)
        inside_count = sum(
            1 for k in range(K)
            if (vae_samples[i, k] - conformal_centers[i]) @ 
               np.linalg.pinv(conformal_cov) @ 
               (vae_samples[i, k] - conformal_centers[i]) <= threshold_24d
        )
        joint_agreements[i] = inside_count / K
        
        # Per-timestep agreement (2D)
        for t in range(12):
            idx = t * 2
            center_2d = conformal_centers[i, idx:idx+2]
            cov_2d = conformal_cov[idx:idx+2, idx:idx+2]
            cov_inv = np.linalg.pinv(cov_2d)
            
            inside = sum(
                1 for k in range(K)
                if (vae_samples[i, k, idx:idx+2] - center_2d) @ 
                   cov_inv @ 
                   (vae_samples[i, k, idx:idx+2] - center_2d) <= threshold_2d
            )
            timestep_agreements[i, t] = inside / K
    
    print("\n" + "="*70)
    print("AGREEMENT STATISTICS")
    print("="*70)
    print(f"K (VAE samples per trajectory): {K}")
    print(f"Expected coverage (1-α): {(1-alpha)*100:.1f}%")
    print(f"\nPer-Timestep Agreement (2D ellipses):")
    print(f"  Mean across all: {timestep_agreements.mean()*100:.2f}%")
    print(f"  Early steps (1-4): {timestep_agreements[:, :4].mean()*100:.2f}%")
    print(f"  Late steps (9-12): {timestep_agreements[:, -4:].mean()*100:.2f}%")
    
    deviation = (joint_agreements.mean() - (1-alpha)) * 100
    print(f"\nDeviation from expected: {deviation:+.2f}%")
    
    if abs(deviation) < 5:
        print("✓ VAE and conformal uncertainties are well-aligned!")
    elif deviation > 5:
        print("⚠ VAE samples more dispersed than conformal (overestimates uncertainty)")
    else:
        print("⚠ VAE samples more concentrated than conformal (underestimates uncertainty)")
    
    print("="*70)
    
    return joint_agreements, timestep_agreements


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_trajectory_metrics(vae_samples, conformal_center, ground_truth, K):
    """Compute minADE, minFDE for VAE and ADE, FDE for deterministic."""
    # Reshape trajectories
    vae_trajs = vae_samples.reshape(K, 12, 2)  # (K, 12, 2)
    det_traj = conformal_center.reshape(12, 2)  # (12, 2)
    gt_traj = ground_truth.reshape(12, 2)      # (12, 2)
    
    # VAE stochastic metrics (minADE, minFDE)
    vae_ades = []
    vae_fdes = []
    for k in range(K):
        # ADE: average displacement across all timesteps
        distances = np.linalg.norm(vae_trajs[k] - gt_traj, axis=1)
        ade = np.mean(distances)
        vae_ades.append(ade)
        
        # FDE: final displacement
        fde = np.linalg.norm(vae_trajs[k, -1] - gt_traj[-1])
        vae_fdes.append(fde)
    
    minADE = np.min(vae_ades)
    minFDE = np.min(vae_fdes)
    
    # Deterministic metrics (ADE, FDE)
    det_distances = np.linalg.norm(det_traj - gt_traj, axis=1)
    det_ADE = np.mean(det_distances)
    det_FDE = np.linalg.norm(det_traj[-1] - gt_traj[-1])
    
    return minADE, minFDE, det_ADE, det_FDE


# ============================================================================
# STAGE 4: VISUALIZATIONS
# ============================================================================

def visualize_predictions(vae_samples, spci, Y_test, sample_idx, alpha, K):
    """Create prediction visualization (Figure 1)."""
    # Get data
    vae_trajs = vae_samples[sample_idx].reshape(K, 12, 2)
    conformal_center = spci.Ensemble_pred_interval_centers[sample_idx]
    conformal_traj = conformal_center.reshape(12, 2)
    conformal_cov = spci.global_cov
    ground_truth = Y_test[sample_idx].reshape(12, 2)
    
    # Compute metrics
    minADE, minFDE, det_ADE, det_FDE = compute_trajectory_metrics(
        vae_samples[sample_idx], conformal_center, Y_test[sample_idx], K
    )
    
    # Create figure with 2x2 layout (2 plots on top, text below)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    colors = plt.cm.viridis(np.linspace(0, 1, 12))
    threshold_2d = chi2.ppf(1 - alpha, df=2)
    
    # Plot 1: VAE Stochastic Samples
    ax = ax1
    for k in range(K):
        ax.plot(vae_trajs[k, :, 0], vae_trajs[k, :, 1], 'b-', alpha=0.3, linewidth=1)
    ax.scatter(vae_trajs[:, 0, 0], vae_trajs[:, 0, 1], c='green', s=30, alpha=0.5, label='Start points')
    ax.scatter(vae_trajs[:, -1, 0], vae_trajs[:, -1, 1], c='red', s=30, alpha=0.5, label='End points')
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g--', linewidth=2.5, 
            marker='s', markersize=6, label='Ground Truth', zorder=4)
    ax.set_title(f'VAE Stochastic Samples (K={K})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: VAE + Conformal Ellipses + Deterministic + Ground Truth
    ax = ax2
    for k in range(K):
        ax.plot(vae_trajs[k, :, 0], vae_trajs[k, :, 1], 'k-', alpha=0.2, linewidth=0.5)
    
    # Draw conformal ellipses
    for t in range(12):
        idx = t * 2
        center_2d = conformal_center[idx:idx+2]
        cov_2d = conformal_cov[idx:idx+2, idx:idx+2]
        
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        eigvals = np.maximum(eigvals, 1e-8)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(threshold_2d * eigvals[0])
        height = 2 * np.sqrt(threshold_2d * eigvals[1])
        
        ellipse = Ellipse(center_2d, width, height, angle=angle,
                         facecolor=colors[t], alpha=0.2,
                         edgecolor=colors[t], linewidth=2)
        ax.add_patch(ellipse)
    
    # Plot trajectories
    ax.plot(conformal_traj[:, 0], conformal_traj[:, 1], 'r-', linewidth=2.5, 
            marker='o', markersize=6, label='ASTRA Deterministic Prediction', zorder=5)
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g--', linewidth=2.5,
            marker='s', markersize=6, label='Ground Truth', zorder=4)
    
    # Dummy elements for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', alpha=0.4, linewidth=1.5, label='VAE stochastic samples'),
        Line2D([0], [0], color='r', linewidth=2.5, marker='o', label='ASTRA Deterministic Prediction'),
        Line2D([0], [0], color='g', linestyle='--', linewidth=2.5, marker='s', label='Ground Truth'),
        plt.Rectangle((0, 0), 1, 1, fc=colors[6], alpha=0.3, ec=colors[6], label='Conformal ellipses (90%)')
    ]
    
    ax.set_title(f'ASTRA Stochastic (VAE) & Deterministic Predictions\nwith Conformal Ellipses (90%)', 
                 fontsize=12, fontweight='bold')
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Add colorbar for timesteps
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=12))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Timestep', fontsize=10)
    
    # Text panel below
    ax = ax3
    ax.axis('off')
    
    metrics_text = f"""
Sample {sample_idx} Metrics                                                    Configuration
{'='*30}                                                    {'='*20}

Stochastic (VAE):                                                  K Samples: {K}
  minADE: {minADE:.4f}                                                      Coverage: {(1-alpha)*100:.0f}%
  minFDE: {minFDE:.4f}

Deterministic:
  ADE: {det_ADE:.4f}
  FDE: {det_FDE:.4f}
"""
    
    ax.text(0.5, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='center')
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(f'results/figures/vae_conformal_predictions_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return metrics for CSV saving
    return {
        'sample_idx': sample_idx,
        'minADE': minADE,
        'minFDE': minFDE,
        'det_ADE': det_ADE,
        'det_FDE': det_FDE,
        'K_samples': K,
        'coverage_level': (1-alpha)*100
    }


def visualize_agreement(vae_samples, spci, Y_test, sample_idx, alpha, K):
    """Create agreement visualization (Figure 2)."""
    # Get data
    vae_trajs = vae_samples[sample_idx].reshape(K, 12, 2)
    conformal_center = spci.Ensemble_pred_interval_centers[sample_idx]
    conformal_cov = spci.global_cov
    ground_truth = Y_test[sample_idx].reshape(12, 2)
    
    # Compute agreement for this sample
    threshold_2d = chi2.ppf(1 - alpha, df=2)
    timestep_agreement = np.zeros(12)
    for t in range(12):
        idx = t * 2
        center_2d = conformal_center[idx:idx+2]
        cov_2d = conformal_cov[idx:idx+2, idx:idx+2]
        cov_inv = np.linalg.pinv(cov_2d)
        
        inside = sum(1 for k in range(K) 
                    if (vae_samples[sample_idx, k, idx:idx+2] - center_2d) @ 
                       cov_inv @ 
                       (vae_samples[sample_idx, k, idx:idx+2] - center_2d) <= threshold_2d)
        timestep_agreement[t] = inside / K
    
    # Create figure with 2x2 layout (2 plots on top, text below)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    colors = plt.cm.viridis(np.linspace(0, 1, 12))
    
    # Plot 1: Timestep 6 detailed view
    ax = ax1
    t = 5  # Timestep 6 (0-indexed)
    idx = t * 2
    center_2d = conformal_center[idx:idx+2]
    cov_2d = conformal_cov[idx:idx+2, idx:idx+2]
    cov_inv = np.linalg.pinv(cov_2d)
    
    inside_mask = np.array([
        (vae_samples[sample_idx, k, idx:idx+2] - center_2d) @ 
        cov_inv @ 
        (vae_samples[sample_idx, k, idx:idx+2] - center_2d) <= threshold_2d
        for k in range(K)
    ])
    
    vae_points = vae_trajs[:, t, :]
    ax.scatter(vae_points[inside_mask, 0], vae_points[inside_mask, 1],
              c='green', s=80, alpha=0.6, edgecolors='darkgreen', linewidths=2,
              label=f'VAE samples inside ({inside_mask.sum()})')
    ax.scatter(vae_points[~inside_mask, 0], vae_points[~inside_mask, 1],
              c='red', s=80, alpha=0.6, edgecolors='darkred', linewidths=2,
              label=f'VAE samples outside ({K - inside_mask.sum()})')
    
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.maximum(eigvals, 1e-8)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * np.sqrt(threshold_2d * eigvals[0])
    height = 2 * np.sqrt(threshold_2d * eigvals[1])
    
    ellipse = Ellipse(center_2d, width, height, angle=angle,
                     facecolor='blue', alpha=0.15, edgecolor='blue', linewidth=3,
                     label='Conformal ellipse (90%)')
    ax.add_patch(ellipse)
    ax.plot(center_2d[0], center_2d[1], 'b*', markersize=20, 
            label='Ellipse center')
    ax.plot(ground_truth[t, 0], ground_truth[t, 1], 'go', markersize=14,
           markeredgecolor='black', markeredgewidth=2.5, label='Ground truth')
    
    ax.set_title(f'Timestep 6: Agreement = {timestep_agreement[t]*100:.1f}%', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Per-timestep agreement
    ax = ax2
    bars = ax.bar(range(1, 13), timestep_agreement, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=1)
    ax.axhline(y=1-alpha, color='r', linestyle='--', linewidth=2,
              label=f'Expected: {(1-alpha)*100:.0f}%')
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Agreement', fontsize=11)
    ax.set_title('Per-Timestep Agreement', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Text panel below
    ax = ax3
    ax.axis('off')
    
    stats_text = f"""
Sample {sample_idx} Statistics                                                    Configuration
{'='*30}                                                    {'='*20}

Per-Timestep Agreement:                                            K Samples: {K}
  Mean:  {timestep_agreement.mean()*100:.1f}%                                                      Coverage: {(1-alpha)*100:.0f}%
  Min:   {timestep_agreement.min()*100:.1f}%
  Max:   {timestep_agreement.max()*100:.1f}%
"""
    
    ax.text(0.5, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='center')
    
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(f'results/figures/vae_conformal_agreement_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return statistics for CSV saving
    return {
        'sample_idx': sample_idx,
        'K_samples': K,
        'coverage_level': (1-alpha)*100,
        'agreement_mean': timestep_agreement.mean()*100,
        'agreement_min': timestep_agreement.min()*100,
        'agreement_max': timestep_agreement.max()*100
    }


def plot_aggregate(joint_agreements, timestep_agreements, alpha, K):
    """Plot aggregate agreement statistics across all samples."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Per-timestep mean agreement
    mean_agreement = timestep_agreements.mean(axis=0)
    std_agreement = timestep_agreements.std(axis=0)
    ax.errorbar(range(1, 13), mean_agreement, yerr=std_agreement, fmt='o-',
               linewidth=2, markersize=8, capsize=5, color='steelblue', label='Mean Agreement')
    ax.axhline(1-alpha, color='red', linestyle='--', linewidth=2, label=f'Expected: {(1-alpha)*100:.0f}%')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Mean Agreement', fontsize=12)
    ax.set_title('Mean Per-Timestep Agreement Across All Test Samples', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/vae_conformal_aggregate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved aggregate plot to: results/figures/vae_conformal_aggregate.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete experimental pipeline."""
    # Configuration
    config_path = './configs/eth.yaml'
    config_path_vae = './configs/eth_vae.yaml'
    subset = 'eth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alpha = 0.1  # 90% confidence
    n_viz = 3    # Number of individual visualizations
    
    print("="*70)
    print("COMPLETE EXPERIMENT: VAE vs CONFORMAL PREDICTION")
    print("="*70)
    print(f"Dataset: {subset}")
    print(f"Device: {device}")
    print(f"Confidence level: {(1-alpha)*100:.0f}%")
    print("="*70)
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    cfg.SUBSET = subset
    cfg.device = device
    cfg.device_list = [device] if device != 'cpu' else []
    set_seed(cfg.TRAIN.SEED)
    
    # Load data
    print("\nLoading data...")
    X_train, Y_train, X_test, Y_test = prepare_real_data(cfg, subset=subset)
    print(f"✓ Train: {len(Y_train)}, Test: {len(Y_test)}")
    
    # Initialize deterministic ASTRA
    print("\nInitializing deterministic ASTRA wrapper...")
    astra_wrapper = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=f'./pretrained_astra_weights/{subset}_best_model.pth',
        unet_weights_path='./pretrained_unet_weights/eth_unet_model_best.pt',
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    astra_wrapper.fit(X_train, Y_train)
    print("✓ ASTRA initialized with pretrained weights")
    
    # STAGE 1: Conformal Prediction
    spci = compute_conformal_regions(X_train, Y_train, X_test, Y_test, 
                                     astra_wrapper, alpha)
    
    # STAGE 2: VAE Samples
    vae_samples, K = generate_vae_samples(
        X_test, config_path_vae,
        f'./pretrained_astra_weights/{subset}_best_model_vae.pth',
        './pretrained_unet_weights/eth_unet_model_best.pt',
        device, subset
    )
    
    # STAGE 3: Agreement Metrics
    joint_agreements, timestep_agreements = compute_agreement_metrics(
        vae_samples, spci, alpha
    )
    
    # STAGE 4: Visualizations
    print("\n" + "="*70)
    print("STAGE 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Collect statistics for CSV export
    prediction_metrics = []
    agreement_statistics = []
    
    for i in range(min(n_viz, len(Y_test))):
        print(f"Creating visualizations for sample {i}...")
        metrics = visualize_predictions(vae_samples, spci, Y_test, i, alpha, K)
        stats = visualize_agreement(vae_samples, spci, Y_test, i, alpha, K)
        prediction_metrics.append(metrics)
        agreement_statistics.append(stats)
    
    # Save statistics to CSV files
    os.makedirs('results/csvs', exist_ok=True)
    
    metrics_df = pd.DataFrame(prediction_metrics)
    metrics_df.to_csv('results/csvs/vae_conformal_predictions_metrics.csv', index=False)
    print(f"\n✓ Saved prediction metrics to: results/csvs/vae_conformal_predictions_metrics.csv")
    
    stats_df = pd.DataFrame(agreement_statistics)
    stats_df.to_csv('results/csvs/vae_conformal_agreement_statistics.csv', index=False)
    print(f"✓ Saved agreement statistics to: results/csvs/vae_conformal_agreement_statistics.csv")
    print(f"✓ Saved {n_viz} figure pairs to: results/figures/")
    
    # Generate aggregate plot
    print("\nCreating aggregate plot...")
    plot_aggregate(joint_agreements, timestep_agreements, alpha, K)
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  PNG visualizations:")
    print(f"    - results/figures/vae_conformal_predictions_sample_0.png ... sample_{n_viz-1}.png")
    print(f"    - results/figures/vae_conformal_agreement_sample_0.png ... sample_{n_viz-1}.png")
    print(f"    - results/figures/vae_conformal_aggregate.png")
    print(f"  CSV statistics:")
    print(f"    - results/csvs/vae_conformal_predictions_metrics.csv")
    print(f"    - results/csvs/vae_conformal_agreement_statistics.csv")
    print("="*70)


if __name__ == "__main__":
    main()