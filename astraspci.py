"""
Complete integration of ASTRA with MultiDimSPCI for trajectory prediction
with uncertainty quantification using conformal prediction.
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

# Import your wrapper (assuming it's saved as astra_wrapper.py)
from astra_wrapper import ASTRASklearnWrapper


def prepare_real_data(cfg, subset='eth', split_ratio=0.8):
    """
    Load and prepare real ETH-UCY data for MultiDimSPCI.
    
    Parameters:
    -----------
    cfg : munch object
        Configuration object
    subset : str
        Which subset to use ('eth', 'hotel', 'univ', 'zara01', 'zara02')
    split_ratio : float
        Train/test split ratio
    
    Returns:
    --------
    X_train, Y_train, X_test, Y_test : dict
        Training and test data with trajectories and images
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
        batch_size=1,  # Process one at a time for simplicity
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
        
        # Convert to numpy and store
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
    
    # Flatten Y for MultiDimSPCI (it expects 2D arrays)
    Y_train_flat = Y_train.reshape(len(Y_train), -1)
    Y_test_flat = Y_test.reshape(len(Y_test), -1)
    
    print(f"\nData shapes:")
    print(f"X_train past_trajectories: {X_train['past_trajectories'].shape}")
    print(f"X_train images: {X_train['images'].shape}")
    print(f"Y_train: {Y_train_flat.shape}")
    print(f"X_test past_trajectories: {X_test['past_trajectories'].shape}")
    print(f"Y_test: {Y_test_flat.shape}")
    
    return X_train, Y_train_flat, X_test, Y_test_flat


def main(B=50):
    """
    Main function to run ASTRA with MultiDimSPCI for conformal prediction.
    
    Parameters:
    -----------
    B : int
        Number of bootstrap samples (default 50, can be reduced for faster testing)
    """
    # Configuration
    config_path = './configs/eth.yaml'
    subset = 'eth'  # Change to 'hotel', 'univ', 'zara01', 'zara02' as needed
    use_unet = True  # Set to False to disable scene features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = munch.munchify(cfg)
    
    # Update config with subset
    cfg.SUBSET = subset
    
    # Set device
    cfg.device = device
    cfg.device_list = [device] if device != 'cpu' else []
    
    # Set seed for reproducibility
    set_seed(cfg.TRAIN.SEED)
    
    print("="*60)
    print("ASTRA + MultiDimSPCI Integration")
    print("="*60)
    print(f"Dataset: {cfg.DATASET}")
    print(f"Subset: {subset}")
    print(f"Using U-Net features: {use_unet}")
    print(f"Device: {device}")
    print(f"Bootstrap samples: {B}")
    print("="*60)
    
    # Step 1: Initialize ASTRA wrapper
    print("\n1. Initializing ASTRA wrapper...")
    astra_wrapper = ASTRASklearnWrapper(
        config_path=config_path,
        pretrained_weights_path=f'./pretrained_astra_weights/{subset}_best_model.pth',
        unet_weights_path='./pretrained_unet_weights/eth_unet_model_best.pt',
        use_pretrained_unet=use_unet,
        device=device,
        dataset='ETH_UCY'
    )
    print("   ✓ ASTRA wrapper initialized")
    
    # Step 2: Load and prepare data
    print("\n2. Loading real data from ETH dataset...")
    X_train, Y_train, X_test, Y_test = prepare_real_data(cfg, subset=subset)
    print("   ✓ Data loaded and prepared")
    
    # Step 3: Fit the wrapper (stores statistics)
    print("\n3. Fitting wrapper (storing data statistics)...")
    astra_wrapper.fit(X_train, Y_train)
    print("   ✓ Wrapper fitted")
    
    # Step 4: Create MultiDimSPCI instance
    print("\n4. Initializing MultiDimSPCI...")
    
    # SPCI expects 2D arrays (n_samples, n_features). prepare_real_data returned
    # dicts with 'past_trajectories' shaped (n, agents, obs_len, dim).
    # Flatten past trajectories for SPCI; keep the dicts for wrapper usage.
    X_train_flat = X_train['past_trajectories'].reshape(len(X_train['past_trajectories']), -1)
    X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
    
    spci = SPCI_and_EnbPI(
        X_train=X_train_flat,
        X_predict=X_test_flat,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=astra_wrapper
    )
    
    print("   ✓ MultiDimSPCI initialized")
    
    # Step 5: Fit bootstrap models for conformal prediction
    print("\n5. Fitting bootstrap models for conformal prediction...")
    if B < 20:
        print(f"   ⚠ Warning: Using only {B} bootstrap samples. Recommended minimum is 20-30.")
        print("   Lower B values may lead to unstable uncertainty estimates.")
    print("   This may take a while...")
    stride = 1  # Stride for multi-step prediction
    spci.fit_bootstrap_models_online_multistep(B=B, stride=stride)
    print(f"   ✓ Fitted {B} bootstrap models")
    
    # Step 6: Compute widths and prediction intervals
    print("\n6. Computing prediction intervals...")
    alpha = 0.1  # Significance level (90% confidence)
    
    # Compute the widths using Ensemble method
    spci.compute_Widths_Ensemble_online(
        alpha=alpha, 
        stride=stride,
        smallT=False,
        past_window=100,
        use_SPCI=False
    )
    
    # Get the results (coverage and size)
    coverage, avg_size = spci.get_results()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Target coverage: {100*(1-alpha):.1f}%")
    print(f"Actual coverage: {100*coverage:.2f}%")
    print(f"Average ellipsoid volume: {avg_size:.2e}")
    print("="*60)
    
    # Step 7: Get predictions with uncertainty for specific samples
    print("\n7. Example predictions with uncertainty...")
    n_examples = 3
    for i in range(min(n_examples, len(X_test['past_trajectories']))):
        # Get single sample
        X_single = {
            'past_trajectories': X_test['past_trajectories'][i:i+1],
            'images': X_test['images'][i:i+1],
            'num_valid': X_test['num_valid'][i:i+1]
        }
        
        # Get point prediction
        point_pred = astra_wrapper.predict(X_single)
        
        # Get prediction interval from MultiDimSPCI
        # Note: You'll need to extract the interval from spci results
        
        print(f"\nSample {i+1}:")
        print(f"  Point prediction shape: {point_pred.shape}")
        print(f"  True future shape: {Y_test[i:i+1].shape}")
        
        # Calculate simple error metrics
        error = np.mean(np.abs(point_pred - Y_test[i:i+1]))
        print(f"  Mean Absolute Error: {error:.4f}")
    
    print("\n" + "="*60)
    print("Integration complete! ASTRA is now working with MultiDimSPCI")
    print("for uncertainty-aware trajectory prediction.")
    print("="*60)
    
    return spci, astra_wrapper


def simple_example(B=10):
    """
    Simpler example with synthetic data - good for testing.
    
    Parameters:
    -----------
    B : int
        Number of bootstrap samples (default 10 for quick testing)
    """
    print("Running simple example with synthetic data...")
    print(f"Using {B} bootstrap samples")
    
    # Initialize wrapper without U-Net for simplicity
    wrapper = ASTRASklearnWrapper(
        config_path='./configs/eth.yaml',
        use_pretrained_unet=False,  # No images needed
        device='cpu'  # Use CPU for testing
    )
    
    # Create synthetic data
    n_train = 100
    n_test = 20
    n_agents = 1
    obs_len = 8
    pred_len = 12
    dim = 2
    
    # Generate synthetic trajectories
    X_train = np.random.randn(n_train, n_agents * obs_len * dim)
    Y_train = np.random.randn(n_train, n_agents * pred_len * dim)
    X_test = np.random.randn(n_test, n_agents * obs_len * dim)
    Y_test = np.random.randn(n_test, n_agents * pred_len * dim)
    
    # Fit wrapper
    wrapper.fit(X_train, Y_train)
    
    # Create MultiDimSPCI
    spci = SPCI_and_EnbPI(
        X_train=X_train,
        X_predict=X_test,
        Y_train=Y_train,
        Y_predict=Y_test,
        fit_func=wrapper
    )
    
    # Fit conformal prediction
    spci.fit_bootstrap_models_online_multistep(B=B, stride=1)
    
    # Compute widths
    spci.compute_Widths_Ensemble_online(
        alpha=0.1,
        stride=1,
        smallT=False,
        past_window=50
    )
    
    # Get results
    coverage, avg_size = spci.get_results()
    
    print(f"\nSimple Example Results:")
    print(f"Coverage: {100*coverage:.2f}%")
    print(f"Average size: {avg_size:.2e}")
    
    return spci, wrapper


if __name__ == "__main__":
    # Choose which example to run
    use_real_data = True  # Set to False for synthetic data test
    
    # Number of bootstrap samples
    # - Use 50+ for production (better uncertainty estimates)
    # - Use 10-20 for quick testing (faster but less stable)
    # - Minimum ~5 for it to work at all
    B = 5  # Change this to reduce computation time
    
    if use_real_data:
        # Run with real ETH data
        spci, wrapper = main(B=B)
    else:
        # Run simple test with synthetic data
        spci, wrapper = simple_example(B=B)