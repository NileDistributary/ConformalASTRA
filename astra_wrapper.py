import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import yaml
import munch
from models.astra_model import ASTRA_model
from models.keypoint_model import UNETEmbeddingExtractor
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ASTRASklearnWrapper(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for ASTRA pretrained model.
    This wrapper makes ASTRA compatible with MultiDimSPCI's expected interface.
    
    Expected input format for X in fit/predict:
    - Dictionary with keys:
        - 'past_trajectories': np.array of shape (n_samples, n_agents, obs_len, 2 or 4)
        - 'images': np.array of shape (n_samples, n_agents, obs_len, H, W, 3) or None
        - 'num_valid': np.array of shape (n_samples,) indicating valid agents
    
    Or if using simplified format:
    - np.array of shape (n_samples, n_features) where features are flattened trajectories
    """
    
    def __init__(self, 
                 config_path='./configs/eth.yaml',
                 pretrained_weights_path=None,
                 unet_weights_path='./pretrained_unet_weights/eth_unet_model_best.pt',
                 use_pretrained_unet=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 dataset='ETH_UCY'):
        """
        Initialize the ASTRA wrapper.
        
        Parameters:
        -----------
        config_path : str
            Path to the ASTRA configuration file
        pretrained_weights_path : str or None
            Path to pretrained ASTRA weights. If None, uses default based on config
        unet_weights_path : str
            Path to pretrained U-Net weights
        use_pretrained_unet : bool or None
            Whether to use pretrained U-Net embedding extractor.
            If None, will be determined from config file.
        device : str
            Device to run the model on ('cuda' or 'cpu')
        dataset : str
            Dataset type ('ETH_UCY' or 'PIE')
        """
        self.config_path = config_path
        self.pretrained_weights_path = pretrained_weights_path
        self.unet_weights_path = unet_weights_path
        self.device = device
        self.dataset = dataset
        
        # Load configuration
        self.cfg = self._load_config()
        
        # Determine whether to use U-Net
        if use_pretrained_unet is None:
            # Use config setting
            self.use_pretrained_unet = self.cfg.MODEL.USE_PRETRAINED_UNET
        else:
            # Override config setting
            self.use_pretrained_unet = use_pretrained_unet
            self.cfg.MODEL.USE_PRETRAINED_UNET = use_pretrained_unet
        
        # Set default pretrained path if not provided
        if self.pretrained_weights_path is None:
            subset = self.cfg.SUBSET if hasattr(self.cfg, 'SUBSET') else 'eth'
            if dataset == 'ETH_UCY':
                self.pretrained_weights_path = f'./pretrained_astra_weights/{subset}_best_model.pth'
            else:
                self.pretrained_weights_path = './pretrained_astra_weights/pie_best_model.pth'
        
        # Initialize models
        self.model = None
        self.embedding_extractor = None
        self._initialize_models()
        
        # Setup image transforms
        self._setup_transforms()
        
    def _load_config(self):
        """Load and process the ASTRA configuration file."""
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = munch.munchify(cfg)
        
        # Update device configuration
        cfg.device = self.device
        cfg.device_list = [self.device] if self.device != 'cpu' else []
        
        return cfg
    
    def _setup_transforms(self):
        """Setup image transformation pipeline."""
        reshape_size = self.cfg.DATA.MIN_RESHAPE_SIZE
        mean = self.cfg.DATA.MEAN
        std = self.cfg.DATA.STD
        
        self.transforms = A.Compose([
            A.LongestMaxSize(reshape_size),
            A.PadIfNeeded(reshape_size, reshape_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean, std, max_pixel_value=255.0),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='yx'))
    
    def _initialize_models(self):
        """Initialize the ASTRA model and U-Net extractor."""
        # Build ASTRA model
        self.model = ASTRA_model(self.cfg)
        
        # Load pretrained weights
        if self.pretrained_weights_path:
            checkpoint = torch.load(self.pretrained_weights_path, 
                                  map_location=torch.device(self.device))
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False) # allow for possible missing parameters, debugging step, consider reverting, Nile Edit. 
            print(f"Loaded checkpoint with {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
            if missing:
                print("Missing (first 5):", missing[:5])

        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize U-Net embedding extractor if needed
        if self.use_pretrained_unet and self.cfg.MODEL.USE_PRETRAINED_UNET:
            self.embedding_extractor = UNETEmbeddingExtractor(self.cfg)
            
            # Load U-Net weights
            unet_checkpoint = torch.load(self.unet_weights_path,
                                        map_location=torch.device(self.device))
            self.embedding_extractor.load_state_dict(unet_checkpoint)
            
            # Remove unnecessary parts for inference
            self.embedding_extractor.unet.decoder = nn.Identity()
            self.embedding_extractor.feature_extractor = nn.Identity()
            self.embedding_extractor.seg_head = nn.Identity()
            self.embedding_extractor.branch1 = nn.Identity()
            self.embedding_extractor.branch2 = nn.Identity()
            self.embedding_extractor.regression_head = nn.Identity()
            
            # Freeze parameters
            for param in self.embedding_extractor.parameters():
                param.requires_grad = False
            
            self.embedding_extractor.eval()
            self.embedding_extractor.to(self.device)
    
    def fit(self, X, y, **kwargs):
        """
        Fit method for scikit-learn compatibility.
        
        For a pretrained model, this mainly stores data statistics.
        
        Parameters:
        -----------
        X : dict or array-like
            Training input samples
        y : array-like of shape (n_samples, n_outputs)
            Target values (future trajectories)
        
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Extract trajectories for statistics
        if isinstance(X, dict):
            trajectories = X.get('past_trajectories', X)
            if len(trajectories.shape) > 2:
                # Flatten for statistics
                trajectories = trajectories.reshape(len(trajectories), -1)
        else:
            trajectories = X
        
        # Store training data statistics
        self.X_mean_ = np.mean(trajectories, axis=0)
        self.X_std_ = np.std(trajectories, axis=0) + 1e-8
        
        if y is not None:
            if len(y.shape) > 2:
                y_flat = y.reshape(len(y), -1)
            else:
                y_flat = y
            self.y_mean_ = np.mean(y_flat, axis=0)
            self.y_std_ = np.std(y_flat, axis=0) + 1e-8
        
        return self
    
    def _apply_normalization(self, past_loc):
        """
        Apply trajectory normalization as done in training.
        
        CRITICAL: This is the key missing piece - ASTRA was trained with 
        trajectories normalized by subtracting the mean of past trajectory.
        
        Parameters:
        -----------
        past_loc : torch.Tensor
            Past trajectory tensor of shape (batch, agents, obs_len, 2)
        
        Returns:
        --------
        past_loc_normalized : torch.Tensor
            Normalized past trajectories
        mean_past_traj : torch.Tensor
            Mean trajectory for denormalization
        """
        # Calculate mean trajectory across time dimension (same as training)
        # Shape: (batch, agents, 1, 2) 
        mean_past_traj = past_loc.mean(dim=-2, keepdims=True)
        
        # Normalize past trajectories
        past_loc_normalized = past_loc - mean_past_traj
        
        return past_loc_normalized, mean_past_traj
    
    def _denormalize_predictions(self, predictions, mean_past_traj):
        """
        Denormalize predictions to restore original scale.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Normalized predictions from ASTRA
        mean_past_traj : torch.Tensor
            Mean trajectory used for normalization
        
        Returns:
        --------
        predictions_denormalized : torch.Tensor
            Predictions in original coordinate system
        """
        # Add back the mean trajectory to restore original coordinates
        predictions_denormalized = predictions + mean_past_traj
        return predictions_denormalized
    
    def predict(self, X, return_multiple_samples=False):
        """
        Make predictions using the ASTRA model.
        
        Parameters:
        -----------
        X : dict or array-like
            Input samples to predict. If dict, should contain:
            - 'past_trajectories': trajectories array
            - 'images': optional images array
            - 'num_valid': optional valid agents indicator
            If array, should be flattened trajectories
        return_multiple_samples : bool
            If True and model uses VAE, return multiple trajectory samples
        
        Returns:
        --------
        y_pred : array-like
            Predicted trajectories
            Shape: (n_samples, n_outputs) if return_multiple_samples=False
            Shape: (n_samples, n_samples_generated, n_outputs) if True
        """
        # Prepare inputs
        past_loc, unet_features = self._prepare_inputs(X)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            # Process in batches if needed
            batch_size = self.cfg.TRAIN.BATCH_SIZE
            n_samples = len(past_loc)
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                
                # Get batch
                batch_past = past_loc[i:batch_end]
                batch_unet = unet_features[i:batch_end] if unet_features is not None else None
                
                # CRITICAL: Apply normalization (was missing!)
                batch_past_normalized, mean_past_traj = self._apply_normalization(batch_past)
                
                # Forward pass through ASTRA with normalized inputs
                # Use the same signature as in our working test: (past_loc, None, unet_features, mode)
                out = self.model(batch_past_normalized, None, batch_unet, mode='testing')
                pred_traj_normalized = out[2]  # model_output is the 3rd element
                
                # CRITICAL: Denormalize predictions (was missing!)
                pred_traj = self._denormalize_predictions(pred_traj_normalized, mean_past_traj)
                
                # Convert to numpy and reshape
                pred_np = pred_traj.cpu().numpy()
                
                # Handle different output formats
                if self.cfg.MODEL.USE_VAE and return_multiple_samples:
                    # Shape: (batch, agents, K, pred_len, output_dim)
                    # Reshape to (batch, K, agents*pred_len*output_dim)
                    batch_size_actual = pred_np.shape[0]
                    n_samples_gen = pred_np.shape[2]
                    pred_np = pred_np.transpose(0, 2, 1, 3, 4)
                    pred_np = pred_np.reshape(batch_size_actual, n_samples_gen, -1)
                else:
                    # Take mean across samples if VAE
                    if self.cfg.MODEL.USE_VAE:
                        pred_np = pred_np.mean(axis=2)  # Average over K samples
                    # Shape: (batch, agents, pred_len, output_dim)
                    # Reshape to (batch, agents*pred_len*output_dim)
                    pred_np = pred_np.reshape(len(pred_np), -1)
                
                predictions.append(pred_np)
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Note: We don't apply the sklearn-style normalization here anymore
        # since we're using the ASTRA-specific normalization above
        
        return predictions
    
    def _prepare_inputs(self, X):
        """
        Prepare input data for ASTRA model.
        
        Parameters:
        -----------
        X : dict or array-like
            Raw input data
        
        Returns:
        --------
        past_loc : torch.Tensor
            Past trajectories tensor
        unet_features : torch.Tensor or None
            Extracted scene features
        """
        # Handle dictionary input
        if isinstance(X, dict):
            past_trajectories = X['past_trajectories']
            images = X.get('images', None)
        else:
            # Assume X is flattened trajectories, need to reshape
            # You'll need to adjust these dimensions based on your specific use case
            n_samples = len(X)
            n_agents = 1  # Default for ETH-UCY
            obs_len = 8   # Default observation length for ETH-UCY
            output_dim = 2 if self.dataset == 'ETH_UCY' else 4
            
            # Reshape from (n_samples, features) to (n_samples, agents, obs_len, output_dim)
            expected_features = n_agents * obs_len * output_dim
            if X.shape[1] == expected_features:
                past_trajectories = X.reshape(n_samples, n_agents, obs_len, output_dim)
            else:
                # Try to infer shape - this might need adjustment based on your data
                total_features = X.shape[1]
                # Assume obs_len=8, output_dim=2 for ETH-UCY
                n_agents = total_features // (obs_len * output_dim)
                past_trajectories = X.reshape(n_samples, n_agents, obs_len, output_dim)
            images = None
        
        # Convert to torch tensor
        past_loc = torch.FloatTensor(past_trajectories).to(self.device)
        
        # Extract U-Net features if needed, or create dummy features
        unet_features = None
        if self.use_pretrained_unet and self.cfg.MODEL.USE_PRETRAINED_UNET:
            if self.embedding_extractor is not None and images is not None:
                unet_features = self._extract_unet_features(images, past_loc.shape)
            else:
                # Create dummy U-Net features with correct shape
                # Shape should be (batch, agents, obs_len, feature_dim)
                batch_size = past_loc.shape[0]
                n_agents = past_loc.shape[1]
                obs_len = past_loc.shape[2]
                feature_dim = self.cfg.MODEL.FEATURE_DIM if hasattr(self.cfg.MODEL, 'FEATURE_DIM') else 512
                
                # Create zero features as placeholder
                unet_features = torch.zeros(batch_size, n_agents, obs_len, feature_dim).to(self.device)
                # Only warn once to avoid spam
                if not hasattr(self, '_warned_about_dummy_features'):
                    print(f"Warning: No images provided but U-Net features expected. Using zero features of shape {unet_features.shape}")
                    self._warned_about_dummy_features = True
        
        return past_loc, unet_features
    

    def _extract_unet_features(self, images, traj_shape):
        """
        Robust U-Net feature extraction.

        Accepts images as numpy or torch in shapes:
         - (B, A, F, H, W, C)  (HWC per image)
         - (B, A, F, C, H, W)  (CHW per image)
         - (B, F, C, H, W)
         - (B, C, H, W)

        Returns:
         - features shaped (B, A, F, feat_dim) on self.device (torch.float32)
        """
        import torch.nn.functional as F

        # Convert to torch tensor on device
        if isinstance(images, np.ndarray):
            imgs = torch.from_numpy(images)
        elif torch.is_tensor(images):
            imgs = images
        else:
            raise ValueError("Unsupported images type for _extract_unet_features")

        imgs = imgs.to(self.device)

        # Normalize HWC -> CHW when needed and capture original dims
        orig_ndim = imgs.ndim
        if orig_ndim == 4:
            # (B, C, H, W) already OK
            B, C, H, W = imgs.shape
            A = 1; F_frames = 1
            flattened = imgs
        elif orig_ndim == 5:
            # could be (B, F, C, H, W) or (B, H, W, C)
            if imgs.shape[1] in (1, 3) or imgs.shape[1] >= 3 and imgs.shape[1] <= 4:
                # assume (B, C, H, W)
                B, C, H, W = imgs.shape
                A = 1; F_frames = 1
                flattened = imgs
            else:
                # (B, F, C, H, W)
                B, F_frames, C, H, W = imgs.shape
                A = 1
                flattened = imgs.reshape(B * F_frames, C, H, W)
        elif orig_ndim == 6:
            # (B, A, F, H, W, C) or (B, A, F, C, H, W)
            B, A, F_frames = imgs.shape[:3]
            if imgs.shape[5] == 3 or imgs.shape[5] == 1:
                # HWC -> convert to CHW
                imgs = imgs.permute(0, 1, 2, 5, 3, 4)  # (B,A,F,C,H,W)
            # now (B,A,F,C,H,W)
            B, A, F_frames, C, H, W = imgs.shape
            flattened = imgs.reshape(B * A * F_frames, C, H, W)
        else:
            raise ValueError(f"Unexpected image tensor dimensionality: {imgs.shape}")

        # If inputs came as HWC numpy inside loop previously (albumentations), ensure dtype float
        flattened = flattened.float()

        # Run extractor in eval
        self.embedding_extractor.eval()
        with torch.no_grad():
            out = self.embedding_extractor(flattened)

        # Handle various extractor outputs (tuple/list or tensor)
        if isinstance(out, (list, tuple)):
            # prefer third element if present, else last
            feat = out[2] if len(out) >= 3 else out[-1]
        else:
            feat = out

        # feat expected shape (N, C) or (N, C, h, w)
        if feat.ndim == 4:
            # global pool spatial dims -> (N, C)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        elif feat.ndim > 2:
            feat = feat.view(feat.size(0), -1)

        N = feat.size(0)
        feat_dim = feat.size(1)

        # reshape back to (B, A, F, feat_dim)
        if orig_ndim == 4:
            feat = feat.view(B, 1, 1, feat_dim)
        elif orig_ndim == 5:
            # was (B, F, C, H, W) -> we flattened (B*F)
            feat = feat.view(B, F_frames, feat_dim).unsqueeze(1)  # (B,1,F,feat)
        else:  # orig_ndim == 6
            feat = feat.view(B, A, F_frames, feat_dim)

        return feat.to(dtype=torch.float32, device=self.device)

    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'config_path': self.config_path,
            'pretrained_weights_path': self.pretrained_weights_path,
            'unet_weights_path': self.unet_weights_path,
            'use_pretrained_unet': self.use_pretrained_unet,
            'device': self.device,
            'dataset': self.dataset
        }
    
    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # Reinitialize models if parameters changed
        self.cfg = self._load_config()
        self._initialize_models()
        self._setup_transforms()
        
        return self