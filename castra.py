"""
ASTRA + MultiDimSPCI: Corrected Implementation
===============================================
Following the actual Algorithm 1 from the MultiDimSPCI paper:
- Uses FIXED pretrained ASTRA (no bootstrap ensemble)
- Computes residuals on calibration set
- Uses quantile regression on nonconformity scores
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
ic.disable()

from astra_wrapper import ASTRASklearnWrapper


# ============================================================================
# DATA LOADING
# ============================================================================

def prepare_real_data(cfg, subset='eth', split_ratio=0.90):
    """
    Load and prepare real ETH-UCY data for MultiDimSPCI.
    """
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
# VISUALIZATION
# ============================================================================

def plot_with_trajectory_uncertainty(spci, X_data, Y_data, astra_wrapper, 
                                    sample_idx=0, confidence=0.9):
    """
    Visualization using covariance from calibration residuals.
    """
    # Get predictions from SPCI's stored predictions (not re-predicting!)
    # This ensures we use the exact same predictions that were used for residuals
    all_preds = spci.Ensemble_pred_interval_centers
    pred_traj = all_preds[sample_idx].reshape(-1, 2)
    true_traj = Y_data[sample_idx].reshape(-1, 2)
    
    print(f"\nVisualizing sample {sample_idx}:")
    print(f"  Predicted trajectory shape: {pred_traj.shape}")
    print(f"  First 3 points: {pred_traj[:3]}")
    print(f"  Are all centers the same? {np.allclose(pred_traj[0], pred_traj[1])}")
    
    # Get covariance
    if hasattr(spci, 'global_cov'):
        cov_matrix = spci.global_cov
    else:
        print("No covariance found, using identity")
        cov_matrix = np.eye(24)
    
    # Get quantile
    if hasattr(spci, 'width_right'):
        q_alpha = np.mean(spci.width_right)
    else:
        q_alpha = np.quantile(spci.test_et, 1 - (1-confidence)/2)
    
    chi2_val = chi2.ppf(confidence, df=2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "r.-", 
            label="Prediction (Centre)", linewidth=2, markersize=8, alpha=0.8)
    ax.plot(true_traj[:, 0], true_traj[:, 1], "g.-", 
            label="Ground Truth", linewidth=2, markersize=8, alpha=0.8)
    
    # Color gradient for ellipses
    colors = plt.cm.viridis(np.linspace(0, 1, len(pred_traj)))
    
    # Plot ellipses
    for t in range(len(pred_traj)):
        idx = t * 2
        cov_2d = cov_matrix[idx:idx+2, idx:idx+2]
        
        # Scale by quantile
        scaled_cov = (q_alpha**2) * cov_2d
        
        eigvals, eigvecs = np.linalg.eigh(scaled_cov)
        eigvals = np.maximum(eigvals, 1e-8)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigvals[0])
        height = 2 * np.sqrt(chi2_val * eigvals[1])
        
        ellipse = Ellipse(pred_traj[t], width, height, angle=angle,
                         facecolor=colors[t], alpha=0.3, 
                         edgecolor=colors[t], linewidth=1)
        ax.add_patch(ellipse)
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_title(f"Sample {sample_idx} - MultiDimSPCI Uncertainty\n"
                f"{int(confidence*100)}% Confidence Ellipses", fontsize=14)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                               norm=plt.Normalize(vmin=1, vmax=len(pred_traj)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Time Step', fontsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config_path = './configs/eth.yaml'
    subset = 'eth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    cfg.SUBSET = subset
    cfg.device = device
    cfg.device_list = [device] if device != 'cpu' else []
    
    set_seed(cfg.TRAIN.SEED)
    
    print("="*70)
    print("ASTRA + MultiDimSPCI: Corrected Implementation")
    print("="*70)
    print(f"Dataset: ETH-UCY, Subset: {subset}")
    print(f"Device: {device}")
    print("Method: MultiDimSPCI with FIXED pretrained ASTRA")
    print("="*70)
    
    # Step 1: Load data
    print("\nStep 1: Loading data from holdout set...")
    X_train, Y_train, X_test, Y_test = prepare_real_data(cfg, subset=subset)
    
    # Step 2: Initialize ASTRA wrapper
    print("\nStep 2: Initializing ASTRA wrapper...")
    astra_wrapper = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=f'./pretrained_astra_weights/{subset}_best_model.pth',
        unet_weights_path='./pretrained_unet_weights/eth_unet_model_best.pt',
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    
    # Fit wrapper (doesn't retrain ASTRA, just sets up interface)
    astra_wrapper.fit(X_train, Y_train)
    print("✓ ASTRA wrapper initialized with pretrained weights")
    
    # Step 3: Initialize SPCI
    print("\nStep 3: Initializing MultiDimSPCI...")
    X_train_flat = X_train['past_trajectories'].reshape(len(X_train['past_trajectories']), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=astra_wrapper  # Not used for predictions in this version
    )

    #Set the rho value for the low rank approximation 
    #spci.r = 17
    print("✓ MultiDimSPCI initialized")
    
    # ========================================================================
    # Step 4: ACTUAL MULTIDIMSPCI - Compute residuals with FIXED predictor
    # ========================================================================
    print("\n" + "="*70)
    print("Step 4: Computing residuals with FIXED predictor (NO bootstrap)")
    print("="*70)
    
    # Get predictions from FIXED pretrained ASTRA
    print("Computing predictions on calibration set...")
    Y_pred_calib = astra_wrapper.predict(X_train)
    print("Computing predictions on test set...")
    Y_pred_test = astra_wrapper.predict(X_test)
    
    # Compute residuals (Algorithm 1, Step 1)
    residuals_calib = Y_train - Y_pred_calib
    residuals_test = Y_test - Y_pred_test
    
    n_calib = len(residuals_calib)
    n_test = len(residuals_test)
    
    print(f"\n✓ Predictions computed with single fixed model")
    print(f"  Calibration samples: {n_calib}")
    print(f"  Test samples: {n_test}")
    
    # Populate SPCI's internal state (what bootstrap would have filled)
    spci.Ensemble_train_interval_centers = Y_pred_calib
    spci.Ensemble_pred_interval_centers = Y_pred_test
    spci.Ensemble_online_resid[:n_calib] = residuals_calib
    spci.Ensemble_online_resid[n_calib:n_calib+n_test] = residuals_test
    
    # Compute nonconformity scores (Algorithm 1, Step 2)
    # This computes Mahalanobis distances: e_t = (ε_t - mean)^T Σ^{-1} (ε_t - mean)
    print("\nComputing covariance matrix and nonconformity scores...")
    spci.get_test_et = False
    spci.train_et = spci.get_et(residuals_calib)
    spci.get_test_et = True
    spci.test_et = spci.get_et(residuals_test)
    spci.all_et = np.concatenate([spci.train_et, spci.test_et])
    
    print(f"✓ Nonconformity scores computed")
    print(f"  Calibration scores: min={spci.train_et.min():.4f}, "
          f"mean={spci.train_et.mean():.4f}, max={spci.train_et.max():.4f}")
    print(f"  Test scores: min={spci.test_et.min():.4f}, "
          f"mean={spci.test_et.mean():.4f}, max={spci.test_et.max():.4f}")
    
    if hasattr(spci, 'global_cov'):
        print(f"  Covariance matrix: {spci.global_cov.shape}")
        print(f"  Covariance determinant: {np.linalg.det(spci.global_cov):.2e}")
    
    # ========================================================================
    # Step 5: Sequential prediction with quantile regression
    # ========================================================================
    print("\n" + "="*70)
    print("Step 5: Computing prediction intervals with quantile regression")
    print("="*70)
    print("This implements Algorithm 1, Steps 3-7:")
    print("  FOR each test point t>T:")
    print("    - Fit quantile regression on past scores")
    print("    - Predict next quantile")
    print("    - Update rolling window")
    
    spci.compute_Widths_Ensemble_online(
        alpha=0.1,
        stride=1,
        smallT=False,
        past_window=10,
        use_SPCI=True  # enables quantile regression
    )
    
    print("\n✓ Prediction intervals computed")
    
    # ========================================================================
    # Step 6: Results
    # ========================================================================
    coverage, avg_size = spci.get_results()
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Target coverage: 90.0%")
    print(f"Empirical coverage: {100*coverage:.2f}%")
    print(f"Average ellipsoid volume: {avg_size:.2e}")
    print("="*70)
    
    print("\n" + "="*70)
    print("Step 7: Running visualizations...")
    print("="*70)
    
    # Diagnostic: Check prediction quality
    print("\nDiagnostic - Checking predictions:")
    print(f"  Stored predictions shape: {spci.Ensemble_pred_interval_centers.shape}")
    print(f"  First sample prediction (first 6 values): {spci.Ensemble_pred_interval_centers[0][:6]}")
    print(f"  First sample reshaped to trajectory:")
    first_traj = spci.Ensemble_pred_interval_centers[0].reshape(-1, 2)
    print(f"    Shape: {first_traj.shape}")
    print(f"    First 3 timesteps:\n{first_traj[:3]}")
    print(f"    All same? {np.allclose(first_traj[0], first_traj[-1])}")
    
    
    # Trajectory-specific uncertainty visualization
    for i in range(min(3, len(X_test['past_trajectories']))):
        fig = plot_with_trajectory_uncertainty(
            spci, X_test, Y_test, astra_wrapper, 
            sample_idx=i, confidence=0.9
        )
        plt.savefig(f'multidimspci_trajectory_{i}.svg', bbox_inches='tight')
        plt.show()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nWhat this implementation does:")
    print("  ✓ Uses FIXED pretrained ASTRA (no bootstrap ensemble)")
    print("  ✓ Computes residuals on calibration set")
    print("  ✓ Estimates global covariance matrix")
    print("  ✓ Computes Mahalanobis distance scores")
    print("  ✓ Applies quantile regression (use_SPCI=True)")
    print("  ✓ Sequential prediction with rolling window")
    print("\nThis follows Algorithm 1 from the MultiDimSPCI paper exactly!")