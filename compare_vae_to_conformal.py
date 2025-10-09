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

from astra_wrapper import ASTRASklearnWrapper


# ============================================================================
# DATA LOADING (from your castra.py)
# ============================================================================

def prepare_real_data(cfg, subset='eth', split_ratio=0.90):
    """Load ETH-UCY data and split into calibration/test sets."""
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
        full_dataset, batch_size=1, shuffle=False, 
        pin_memory=True, drop_last=False
    )
    
    # Collect data
    all_past, all_future, all_images, all_num_valid = [], [], [], []
    
    for batch in loader:
        past_loc, fut_loc, num_valid, imgs, _, _ = batch
        all_past.append(past_loc.numpy())
        all_future.append(fut_loc.numpy())
        all_images.append(imgs.numpy())
        all_num_valid.append(num_valid.numpy())
    
    all_past = np.concatenate(all_past, axis=0)
    all_future = np.concatenate(all_future, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_num_valid = np.concatenate(all_num_valid, axis=0)
    
    # Split
    n_samples = len(all_past)
    n_train = int(n_samples * split_ratio)
    
    X_train = {
        'past_trajectories': all_past[:n_train],
        'images': all_images[:n_train],
        'num_valid': all_num_valid[:n_train]
    }
    Y_train = all_future[:n_train].reshape(n_train, -1)
    
    X_test = {
        'past_trajectories': all_past[n_train:],
        'images': all_images[n_train:],
        'num_valid': all_num_valid[n_train:]
    }
    Y_test = all_future[n_train:].reshape(n_samples - n_train, -1)
    
    return X_train, Y_train, X_test, Y_test


# ============================================================================
# STAGE 1: DETERMINISTIC CONFORMAL PREDICTION (from castra.py)
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
        use_SPCI=True  # Enables quantile regression
    )
    
    # Get results
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
    """
    Generate K stochastic samples from ASTRA with VAE enabled.
    This uses a SEPARATE model instance with USE_VAE=True.
    """
    print("\n" + "="*70)
    print("STAGE 2: GENERATING VAE STOCHASTIC SAMPLES")
    print("="*70)
    
    # Load config and enable VAE
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
    
    # Create VAE wrapper
    vae_wrapper = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=pretrained_path,
        unet_weights_path=unet_path,
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    
    # CRITICAL: Override config after initialization
    vae_wrapper.cfg.MODEL.USE_VAE = True
    vae_wrapper.cfg.MODEL.K = K
    
    # Dummy fit (just initializes)
    dummy_X = {
        'past_trajectories': np.zeros((1, 8, 2)),
        'images': np.zeros((1, 224, 224, 3)),
        'num_valid': np.array([8])
    }
    vae_wrapper.fit(dummy_X, np.zeros((1, 24)))
    
    # Generate samples
    print(f"\nGenerating {K} samples for {len(X_test['past_trajectories'])} test trajectories...")
    vae_samples = vae_wrapper.predict(X_test, return_multiple_samples=True)
    
    print(f"✓ VAE samples generated: {vae_samples.shape}")
    print(f"  Expected shape: (n_test, K, 24)")
    
    return vae_samples, K


# ============================================================================
# STAGE 3: COMPUTE COVERAGE METRICS
# ============================================================================

def compute_coverage_metrics(vae_samples, spci, alpha=0.1):
    """
    Compute what % of VAE samples fall within conformal ellipsoids.
    """
    print("\n" + "="*70)
    print("STAGE 3: COMPUTING COVERAGE METRICS")
    print("="*70)
    print("For each test trajectory:")
    print("  1. Extract conformal prediction ellipsoid")
    print("  2. Check which VAE samples fall inside")
    print("  3. Compute coverage percentage")
    
    n_test = len(vae_samples)
    K = vae_samples.shape[1]
    
    # Chi-squared threshold
    dim = 24
    threshold_joint = chi2.ppf(1 - alpha, df=dim)
    threshold_2d = chi2.ppf(1 - alpha, df=2)
    
    joint_coverages = []
    timestep_coverages = []
    
    for i in range(n_test):
        # Get conformal prediction for this sample
        conformal_center = spci.Ensemble_pred_interval_centers[i]
        conformal_cov = spci.global_cov  # Using global covariance
        
        # Get VAE samples
        vae_samples_i = vae_samples[i]  # Shape: (K, 24)
        
        # --- Joint coverage (full 24D ellipsoid) ---
        try:
            cov_inv = np.linalg.inv(conformal_cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(conformal_cov)
        
        inside_joint = 0
        for k in range(K):
            diff = vae_samples_i[k] - conformal_center
            dist_sq = diff @ cov_inv @ diff
            if dist_sq <= threshold_joint:
                inside_joint += 1
        
        joint_coverages.append(inside_joint / K)
        
        # --- Per-timestep coverage (12 independent 2D ellipses) ---
        timestep_cov = np.zeros(12)
        for t in range(12):
            idx = t * 2
            center_2d = conformal_center[idx:idx+2]
            cov_2d = conformal_cov[idx:idx+2, idx:idx+2]
            
            try:
                cov_inv_2d = np.linalg.inv(cov_2d)
            except np.linalg.LinAlgError:
                cov_inv_2d = np.linalg.pinv(cov_2d)
            
            inside = 0
            for k in range(K):
                diff_2d = vae_samples_i[k, idx:idx+2] - center_2d
                dist_sq_2d = diff_2d @ cov_inv_2d @ diff_2d
                if dist_sq_2d <= threshold_2d:
                    inside += 1
            
            timestep_cov[t] = inside / K
        
        timestep_coverages.append(timestep_cov)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_test}...")
    
    joint_coverages = np.array(joint_coverages)
    timestep_coverages = np.array(timestep_coverages)
    
    # Summary statistics
    print("\n" + "="*70)
    print("COVERAGE STATISTICS")
    print("="*70)
    print(f"K (VAE samples per trajectory): {K}")
    print(f"Expected coverage (1-α): {(1-alpha)*100:.1f}%")
    print(f"\nJoint Coverage (24D ellipsoid):")
    print(f"  Mean:   {joint_coverages.mean()*100:.2f}%")
    print(f"  Std:    {joint_coverages.std()*100:.2f}%")
    print(f"  Median: {np.median(joint_coverages)*100:.2f}%")
    print(f"  Range:  [{joint_coverages.min()*100:.1f}%, {joint_coverages.max()*100:.1f}%]")
    print(f"\nPer-Timestep Coverage (2D ellipses):")
    print(f"  Mean across all: {timestep_coverages.mean()*100:.2f}%")
    print(f"  Early steps (1-4): {timestep_coverages[:, :4].mean()*100:.2f}%")
    print(f"  Late steps (9-12): {timestep_coverages[:, -4:].mean()*100:.2f}%")
    
    deviation = (joint_coverages.mean() - (1-alpha)) * 100
    print(f"\nDeviation from expected: {deviation:+.2f}%")
    
    if abs(deviation) < 5:
        print("✓ VAE and conformal uncertainties are well-aligned!")
    elif deviation > 5:
        print("⚠ VAE samples more dispersed than conformal (overestimates uncertainty)")
    else:
        print("⚠ VAE samples more concentrated than conformal (underestimates uncertainty)")
    
    print("="*70)
    
    return joint_coverages, timestep_coverages


# ============================================================================
# STAGE 4: VISUALIZATIONS
# ============================================================================

def visualize_comparison(vae_samples, spci, Y_test, sample_idx, alpha, K):
    """Create comprehensive visualization for a single sample."""
    
    # Get data
    vae_trajs = vae_samples[sample_idx].reshape(K, 12, 2)
    conformal_center = spci.Ensemble_pred_interval_centers[sample_idx]
    conformal_traj = conformal_center.reshape(12, 2)
    conformal_cov = spci.global_cov
    ground_truth = Y_test[sample_idx].reshape(12, 2)
    
    # Compute coverage for this sample
    threshold_2d = chi2.ppf(1 - alpha, df=2)
    timestep_cov = np.zeros(12)
    for t in range(12):
        idx = t * 2
        center_2d = conformal_center[idx:idx+2]
        cov_2d = conformal_cov[idx:idx+2, idx:idx+2]
        cov_inv = np.linalg.pinv(cov_2d)
        
        inside = sum(1 for k in range(K) 
                    if (vae_samples[sample_idx, k, idx:idx+2] - center_2d) @ 
                       cov_inv @ 
                       (vae_samples[sample_idx, k, idx:idx+2] - center_2d) <= threshold_2d)
        timestep_cov[t] = inside / K
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = plt.cm.viridis(np.linspace(0, 1, 12))
    
    # Plot 1: VAE samples only
    ax = axes[0, 0]
    for k in range(K):
        ax.plot(vae_trajs[k, :, 0], vae_trajs[k, :, 1], 'b-', alpha=0.3, linewidth=1)
    ax.scatter(vae_trajs[:, 0, 0], vae_trajs[:, 0, 1], c='green', s=30, alpha=0.5)
    ax.scatter(vae_trajs[:, -1, 0], vae_trajs[:, -1, 1], c='red', s=30, alpha=0.5)
    ax.set_title(f'VAE Stochastic Samples (K={K})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Comparison
    ax = axes[0, 1]
    for k in range(K):
        ax.plot(vae_trajs[k, :, 0], vae_trajs[k, :, 1], 'gray', alpha=0.15, linewidth=0.5)
    ax.plot(conformal_traj[:, 0], conformal_traj[:, 1], 'r-', linewidth=3, 
           marker='o', markersize=6, label='Conformal (Deterministic)', zorder=5)
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'g--', linewidth=2,
           marker='s', markersize=6, label='Ground Truth', zorder=4)
    ax.set_title(f'Sample {sample_idx}: VAE vs Conformal')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: VAE + Conformal ellipses
    ax = axes[0, 2]
    for k in range(K):
        ax.plot(vae_trajs[k, :, 0], vae_trajs[k, :, 1], 'k-', alpha=0.2, linewidth=0.5)
    
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
    
    ax.plot(conformal_traj[:, 0], conformal_traj[:, 1], 'r-', linewidth=2, marker='o')
    ax.set_title(f'VAE + Conformal Ellipses ({int((1-alpha)*100)}%)')
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Plot 4: Detailed timestep
    ax = axes[1, 0]
    t = 5  # Middle timestep
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
              label=f'Inside ({inside_mask.sum()})')
    ax.scatter(vae_points[~inside_mask, 0], vae_points[~inside_mask, 1],
              c='red', s=80, alpha=0.6, edgecolors='darkred', linewidths=2,
              label=f'Outside ({K - inside_mask.sum()})')
    
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    eigvals = np.maximum(eigvals, 1e-8)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * np.sqrt(threshold_2d * eigvals[0])
    height = 2 * np.sqrt(threshold_2d * eigvals[1])
    
    ellipse = Ellipse(center_2d, width, height, angle=angle,
                     facecolor='blue', alpha=0.15, edgecolor='blue', linewidth=3)
    ax.add_patch(ellipse)
    ax.plot(center_2d[0], center_2d[1], 'b*', markersize=20)
    ax.plot(ground_truth[t, 0], ground_truth[t, 1], 'go', markersize=14,
           markeredgecolor='black', markeredgewidth=2)
    
    ax.set_title(f'Timestep {t+1}: Coverage = {timestep_cov[t]*100:.1f}%')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axis('equal')
    
    # Plot 5: Per-timestep coverage
    ax = axes[1, 1]
    bars = ax.bar(range(1, 13), timestep_cov, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=1)
    ax.axhline(y=1-alpha, color='r', linestyle='--', linewidth=2,
              label=f'Expected: {(1-alpha)*100:.0f}%')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Coverage')
    ax.set_title('Per-Timestep Coverage')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 6: Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    joint_cov = np.mean([
        (vae_samples[sample_idx, k] - conformal_center) @ 
        np.linalg.pinv(conformal_cov) @ 
        (vae_samples[sample_idx, k] - conformal_center) <= chi2.ppf(1-alpha, df=24)
        for k in range(K)
    ])
    
    stats_text = f"""
    Sample {sample_idx} Statistics
    {'='*32}
    
    K samples: {K}
    Confidence: {(1-alpha)*100:.0f}%
    
    Joint Coverage:
      {joint_cov*100:.1f}% of VAE samples
      Expected: {(1-alpha)*100:.0f}%
    
    Per-Timestep:
      Mean:  {timestep_cov.mean()*100:.1f}%
      Min:   {timestep_cov.min()*100:.1f}%
      Max:   {timestep_cov.max()*100:.1f}%
    """
    
    ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'vae_conformal_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_aggregate(joint_coverages, timestep_coverages, alpha, K):
    """Plot aggregate statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Joint coverage distribution
    ax = axes[0, 0]
    ax.hist(joint_coverages, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1-alpha, color='red', linestyle='--', linewidth=3,
              label=f'Expected: {(1-alpha)*100:.0f}%')
    ax.axvline(joint_coverages.mean(), color='green', linestyle='-', linewidth=3,
              label=f'Mean: {joint_coverages.mean()*100:.1f}%')
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Joint Coverage Distribution (K={K})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Per-timestep mean
    ax = axes[0, 1]
    mean_cov = timestep_coverages.mean(axis=0)
    std_cov = timestep_coverages.std(axis=0)
    ax.errorbar(range(1, 13), mean_cov, yerr=std_cov, fmt='o-',
               linewidth=2, markersize=8, capsize=5, color='steelblue')
    ax.axhline(1-alpha, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Coverage')
    ax.set_title('Mean Per-Timestep Coverage')
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    
    # Coverage variability
    ax = axes[1, 0]
    ax.scatter(range(len(joint_coverages)), joint_coverages, s=20, alpha=0.5)
    ax.axhline(1-alpha, color='red', linestyle='--', linewidth=2)
    ax.axhline(joint_coverages.mean(), color='green', linestyle='-', alpha=0.7)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage Variability')
    ax.grid(alpha=0.3)
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    deviation = (joint_coverages.mean() - (1-alpha)) * 100
    summary = f"""
    AGGREGATE STATISTICS
    {'='*28}
    
    Samples: {len(joint_coverages)}
    K per sample: {K}
    Confidence: {(1-alpha)*100:.0f}%
    
    JOINT COVERAGE:
      Mean:   {joint_coverages.mean()*100:.2f}%
      Std:    {joint_coverages.std()*100:.2f}%
      Median: {np.median(joint_coverages)*100:.2f}%
      
    Expected: {(1-alpha)*100:.0f}%
    Deviation: {deviation:+.2f}%
    
    TIMESTEP AVERAGE:
      {mean_cov.mean()*100:.2f}%
    """
    ax.text(0.05, 0.5, summary, fontsize=10, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig('vae_conformal_aggregate.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete experimental pipeline in one script."""
    
    # Configuration
    config_path = './configs/eth.yaml'
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
    
    # ========== STAGE 1: Conformal Prediction ==========
    spci = compute_conformal_regions(X_train, Y_train, X_test, Y_test, 
                                     astra_wrapper, alpha)
    
    # ========== STAGE 2: VAE Samples ==========
    vae_samples, K = generate_vae_samples(
        X_test, config_path,
        f'./pretrained_astra_weights/{subset}_best_model.pth',
        './pretrained_unet_weights/eth_unet_model_best.pt',
        device, subset
    )
    
    # ========== STAGE 3: Coverage Metrics ==========
    joint_coverages, timestep_coverages = compute_coverage_metrics(
        vae_samples, spci, alpha
    )
    
    # ========== STAGE 4: Visualizations ==========
    print("\n" + "="*70)
    print("STAGE 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Individual samples
    for i in range(min(n_viz, len(Y_test))):
        print(f"Creating visualization for sample {i}...")
        visualize_comparison(vae_samples, spci, Y_test, i, alpha, K)
    
    # Aggregate
    print("Creating aggregate plots...")
    plot_aggregate(joint_coverages, timestep_coverages, alpha, K)
    
    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nKEY FINDINGS:")
    print(f"  VAE Coverage:  {joint_coverages.mean()*100:.2f}% ± {joint_coverages.std()*100:.2f}%")
    print(f"  Expected:      {(1-alpha)*100:.0f}%")
    print(f"  Deviation:     {(joint_coverages.mean()-(1-alpha))*100:+.2f}%")
    print("\nGenerated files:")
    print(f"  - vae_conformal_sample_0.png ... sample_{n_viz-1}.png")
    print(f"  - vae_conformal_aggregate.png")
    print("="*70)


if __name__ == "__main__":
    main()