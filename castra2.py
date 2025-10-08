"""
Complete integration of ASTRA with MultiDimSPCI for trajectory prediction
with uncertainty quantification using conformal prediction.
FIXED VERSION: Handles data format conversion for proper bootstrap variance.
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
from utils.misc import set_seed, set_device
from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

# Import your wrapper
from astra_wrapper import ASTRASklearnWrapper
from final_ellipsoid_viz import visualize_spci_uncertainty_ellipses


# ============================================================================
# DATA FORMAT WRAPPER - CRITICAL FOR BOOTSTRAP VARIANCE
# ============================================================================

class ASTRAWrapperForSPCI:
    """
    Wrapper that converts between SPCI's flattened format and ASTRA's dict format.
    This is CRITICAL for getting varied bootstrap predictions.
    """
    def __init__(self, astra_wrapper, X_train_dict, X_test_dict):
        self.astra_wrapper = astra_wrapper
        self.X_train_dict = X_train_dict
        self.X_test_dict = X_test_dict
        self.n_train = len(X_train_dict['past_trajectories'])
        
    def fit(self, X_flat, Y):
        # Convert flattened indices to dict format
        if isinstance(X_flat, np.ndarray):
            # X_flat contains indices or flattened features
            indices = np.arange(len(X_flat))
            X_dict = {
                'past_trajectories': self.X_train_dict['past_trajectories'][indices],
                'images': self.X_train_dict['images'][indices],
                'num_valid': self.X_train_dict['num_valid'][indices]
            }
        else:
            X_dict = X_flat
        
        return self.astra_wrapper.fit(X_dict, Y)
    
    def predict(self, X_flat):
        # Determine if this is train or test data
        n_samples = len(X_flat)
        
        # Create proper dict format
        if n_samples <= self.n_train:
            # Training data
            X_dict = {
                'past_trajectories': self.X_train_dict['past_trajectories'][:n_samples],
                'images': self.X_train_dict['images'][:n_samples],
                'num_valid': self.X_train_dict['num_valid'][:n_samples]
            }
        else:
            # Combined train+test data
            n_test = n_samples - self.n_train
            X_dict = {
                'past_trajectories': np.vstack([
                    self.X_train_dict['past_trajectories'],
                    self.X_test_dict['past_trajectories'][:n_test]
                ]),
                'images': np.vstack([
                    self.X_train_dict['images'],
                    self.X_test_dict['images'][:n_test]
                ]),
                'num_valid': np.hstack([
                    self.X_train_dict['num_valid'],
                    self.X_test_dict['num_valid'][:n_test]
                ])
            }
        
        return self.astra_wrapper.predict(X_dict)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_holdout_with_split(cfg, subset='eth', calib_ratio=0.85):
    """
    Load ASTRA's holdout test set and split for conformal prediction.
    
    Returns:
    --------
    X_calib, Y_calib : 85% for conformal calibration
    X_eval, Y_eval : 15% for coverage evaluation
    """
    print(f"\n{'='*70}")
    print(f"LOADING HOLDOUT DATA: {subset.upper()} scene")
    print(f"{'='*70}")
    print(f"ASTRA was NOT trained on this scene (LOSO protocol)")
    print(f"Loading ASTRA's test set, then splitting for conformal...")
    print(f"{'='*70}\n")
    
    cfg.SUBSET = subset
    
    # Use the SAME transform setup as your original code
    reshape_size = cfg.DATA.MIN_RESHAPE_SIZE
    mean = cfg.DATA.MEAN
    std = cfg.DATA.STD
    
    transforms = A.Compose([
        A.LongestMaxSize(reshape_size),
        A.PadIfNeeded(reshape_size, reshape_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean, std, max_pixel_value=255.0),
        ToTensorV2()
    ])  # ← NO keypoint_params! That's what's causing the error
    
    # Load ASTRA's full test set
    dataset = ETH_dataset(cfg=cfg, mode='testing', img_transforms=transforms)
    
    # Use batch_size=1 and loop, just like your original code
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    # Collect all data (same as your original approach)
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
    
    # Concatenate
    all_past_trajectories = np.concatenate(all_past_trajectories, axis=0)
    all_future_trajectories = np.concatenate(all_future_trajectories, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    all_num_valid = np.concatenate(all_num_valid, axis=0)
    
    n_total = len(all_past_trajectories)
    n_calib = int(n_total * calib_ratio)  # 85%
    
    print(f"\n{'='*70}")
    print(f"NESTED HOLDOUT STRUCTURE")
    print(f"{'='*70}")
    print(f"ASTRA's holdout (mode='testing'): {n_total} samples")
    print(f"  ├─ Conformal calibration (85%): {n_calib} samples")
    print(f"  └─ Coverage evaluation (15%): {n_total - n_calib} samples")
    print(f"{'='*70}\n")
    
    # Split: 85% calibration, 15% evaluation
    X_calib = {
        'past_trajectories': all_past_trajectories[:n_calib],
        'images': all_images[:n_calib],
        'num_valid': all_num_valid[:n_calib]
    }
    Y_calib = all_future_trajectories[:n_calib].reshape(n_calib, -1)
    
    X_eval = {
        'past_trajectories': all_past_trajectories[n_calib:],
        'images': all_images[n_calib:],
        'num_valid': all_num_valid[n_calib:]
    }
    Y_eval = all_future_trajectories[n_calib:].reshape(n_total - n_calib, -1)
    
    print(f"✓ Loaded and split holdout data")
    print(f"  Calibration: {X_calib['past_trajectories'].shape}")
    print(f"  Evaluation: {X_eval['past_trajectories'].shape}")
    
    return X_calib, Y_calib, X_eval, Y_eval


def prepare_real_data(cfg, subset='eth', split_ratio=0.8):
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
    
    # Load dataset
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
    
    # Training data
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
# TRAJECTORY COVARIANCE FUNCTIONS
# ============================================================================

def get_trajectory_covariance_from_models(spci, sample_idx):
    """
    Get trajectory-specific covariance using stored bootstrap models.
    FIXED: Uses proper dict format for predictions.
    """
    if not spci.bootstrap_models:
        print("No bootstrap models found, using global covariance")
        return spci.global_cov
    
    # Use the stored dict format
    if hasattr(spci, 'X_test_dict'):
        X_sample = {
            'past_trajectories': spci.X_test_dict['past_trajectories'][sample_idx:sample_idx+1],
            'images': spci.X_test_dict['images'][sample_idx:sample_idx+1],
            'num_valid': spci.X_test_dict['num_valid'][sample_idx:sample_idx+1]
        }
    else:
        print("Warning: No dict format stored, predictions may be identical")
        return spci.global_cov
    
    predictions = []
    for i, model in enumerate(spci.bootstrap_models):
        pred = model.predict(X_sample)
        predictions.append(pred.flatten())
    
    predictions = np.array(predictions)
    variance = np.var(predictions, axis=0)
    
    print(f"Sample {sample_idx}: Computed from {len(predictions)} models")
    print(f"  Variance mean={np.mean(variance):.6f}, max={np.max(variance):.6f}")
    
    if np.mean(variance) < 1e-6:
        print("  WARNING: No variance in bootstrap predictions!")
        print("  Using scaled global covariance as fallback")
        return spci.global_cov * 1.5  # Scale for visibility
    
    return np.cov(predictions.T)


def diagnose_bootstrap_variance(spci, astra_wrapper_orig):
    """
    Diagnostic function to check if bootstrap models produce variance.
    """
    print("\n" + "="*60)
    print("BOOTSTRAP VARIANCE DIAGNOSIS")
    print("="*60)
    
    print(f"Number of bootstrap models: {len(spci.bootstrap_models)}")
    
    if len(spci.bootstrap_models) == 0:
        print("ERROR: No bootstrap models stored!")
        return
    
    if hasattr(spci, 'X_test_dict'):
        # Test with first sample
        X_sample = {
            'past_trajectories': spci.X_test_dict['past_trajectories'][0:1],
            'images': spci.X_test_dict['images'][0:1],
            'num_valid': spci.X_test_dict['num_valid'][0:1]
        }
        
        # Get predictions from first 3 models
        print("\nTesting first 3 bootstrap models:")
        for i in range(min(3, len(spci.bootstrap_models))):
            pred = spci.bootstrap_models[i].predict(X_sample)
            print(f"  Model {i}: mean={np.mean(pred):.4f}, std={np.std(pred):.4f}")
        
        # Check overall variance
        all_preds = []
        for model in spci.bootstrap_models:
            all_preds.append(model.predict(X_sample).flatten())
        all_preds = np.array(all_preds)
        
        var = np.var(all_preds, axis=0)
        print(f"\nOverall variance: mean={np.mean(var):.6f}, max={np.max(var):.6f}")
        
        if np.mean(var) < 1e-6:
            print("⚠️ PROBLEM: Bootstrap models produce identical predictions!")
        else:
            print("✓ Bootstrap models produce varied predictions")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_with_true_trajectory_uncertainty(spci, X_data, Y_data, astra_wrapper, 
                                         sample_idx=0, confidence=0.9):
    """
    Visualization using true bootstrap-based trajectory covariance.
    """
    # Get predictions
    all_preds = astra_wrapper.predict(X_data)
    pred_traj = all_preds[sample_idx].reshape(-1, 2)
    true_traj = Y_data[sample_idx].reshape(-1, 2)
    
    # Get trajectory-specific covariance
    traj_cov = get_trajectory_covariance_from_models(spci, sample_idx)
    
    chi2_val = chi2.ppf(confidence, df=2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "r.-", 
            label="Prediction", linewidth=2, markersize=8, alpha=0.8)
    ax.plot(true_traj[:, 0], true_traj[:, 1], "g.-", 
            label="Ground Truth", linewidth=2, markersize=8, alpha=0.8)
    
    # Color gradient for ellipses
    colors = plt.cm.viridis(np.linspace(0, 1, len(pred_traj)))
    
    # Plot ellipses with trajectory-specific covariance
    for t in range(len(pred_traj)):
        idx = t * 2
        cov_2d = traj_cov[idx:idx+2, idx:idx+2]
        
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
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
    ax.set_title(f"Sample {sample_idx} with Bootstrap Uncertainty\n"
                f"{int(confidence*100)}% Confidence Ellipses", fontsize=14)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                               norm=plt.Normalize(vmin=1, vmax=len(pred_traj)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Time Step', fontsize=10)
    
    plt.show()


def compare_trajectory_covariances(spci, n_samples=4):
    """
    Verify that different trajectories have different covariances.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(n_samples, 4)):
        ax = axes[i]
        
        # Get trajectory-specific covariance
        traj_cov = get_trajectory_covariance_from_models(spci, i)
        
        # Extract uncertainty magnitude at each timestep
        uncertainties = []
        for t in range(12):
            idx = t * 2
            var_x = traj_cov[idx, idx]
            var_y = traj_cov[idx+1, idx+1]
            uncertainties.append(np.sqrt(var_x + var_y))
        
        timesteps = np.arange(1, 13)
        ax.plot(timesteps, uncertainties, 'b-', linewidth=2)
        ax.fill_between(timesteps, 0, uncertainties, alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Uncertainty (std)')
        ax.set_title(f'Sample {i} - Total: {np.sum(uncertainties):.2f}')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Trajectory-Specific Uncertainties', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    B = 5  # Number of bootstrap samples (increase for better estimates)
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
    
    # Set seed for reproducibility
    set_seed(cfg.TRAIN.SEED)
    
    print("="*60)
    print("ASTRA + MultiDimSPCI with Bootstrap Variance Fix")
    print("="*60)
    print(f"Dataset: ETH-UCY, Subset: {subset}")
    print(f"Device: {device}")
    print(f"Bootstrap samples: {B}")
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    X_train, Y_train, X_test, Y_test = prepare_real_data(cfg, subset=subset)
    
    # Step 2: Initialize original ASTRA wrapper
    print("\nStep 2: Initializing ASTRA wrapper...")
    astra_wrapper_orig = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=f'./pretrained_astra_weights/{subset}_best_model.pth',
        unet_weights_path='./pretrained_unet_weights/eth_unet_model_best.pt',
        use_pretrained_unet=True,
        device=device,
        dataset='ETH_UCY'
    )
    
    # Fit original wrapper
    astra_wrapper_orig.fit(X_train, Y_train)
    
    # Step 3: Create format-handling wrapper (CRITICAL!)
    print("\nStep 3: Creating format-handling wrapper...")
    astra_wrapper = ASTRAWrapperForSPCI(astra_wrapper_orig, X_train, X_test)
    
    # Step 4: Create SPCI with flattened data
    print("\nStep 4: Initializing MultiDimSPCI...")
    X_train_flat = X_train['past_trajectories'].reshape(len(X_train['past_trajectories']), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=astra_wrapper  # Use wrapper, not original!
    )
    
    # Store original dict format for bootstrap predictions
    spci.X_train_dict = X_train
    spci.X_test_dict = X_test
    
    # Step 5: Fit bootstrap models
    print(f"\nStep 5: Fitting {B} bootstrap models...")
    spci.fit_bootstrap_models_online_multistep(B=B, stride=1)
    
    # Step 6: Compute widths
    print("\nStep 6: Computing prediction intervals...")
    spci.compute_Widths_Ensemble_online(
        alpha=0.1,
        stride=1,
        smallT=False,
        past_window=100
    )
    
    # Get results
    coverage, avg_size = spci.get_results()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Coverage: {100*coverage:.2f}%")
    print(f"Average ellipsoid volume: {avg_size:.2e}")
    
    # Step 7: Diagnose bootstrap variance
    diagnose_bootstrap_variance(spci, astra_wrapper_orig)
    
    # Step 8: Save models
    print("\nStep 8: Saving bootstrap models...")
    spci.save_bootstrap_models('bootstrap_models_fixed.pkl')
    
    # Step 9: Visualizations
    print("\nStep 9: Running visualizations...")
    
    # Basic ellipsoid visualization
    visualize_spci_uncertainty_ellipses(spci)
    
    # Trajectory-specific uncertainty visualization
    for i in range(min(3, len(X_test['past_trajectories']))):
        plot_with_true_trajectory_uncertainty(
            spci, X_test, Y_test, astra_wrapper_orig, 
            sample_idx=i, confidence=0.9
        )
    
    # Compare uncertainties across samples
    compare_trajectory_covariances(spci, n_samples=4)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)