"""
Baseline: Coordinate-wise Conformal Prediction
Fixed version - corrected quantile forest usage
"""
import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor

class CoordinateWiseCP:
    """
    Coordinate-wise conformal prediction baseline.
    Creates independent prediction intervals for each output coordinate.
    """
    def __init__(self, alpha=0.1, use_quantile_regression=True):
        self.alpha = alpha
        self.use_quantile_regression = use_quantile_regression
        self.quantiles = {}  # Store quantiles per coordinate
        self.qrf_models = {}  # Store QRF models per coordinate
        self.intervals = None
        
    def calibrate(self, Y_calib, Y_pred_calib):
        """
        Calibrate the conformal predictor
        
        Args:
            Y_calib: True values (n_calib, n_coords)
            Y_pred_calib: Predicted values (n_calib, n_coords)
        """
        print(f"  Calibrating on {len(Y_calib)} samples, {Y_calib.shape[1]} coordinates")
        
        residuals = np.abs(Y_calib - Y_pred_calib)
        n_coords = residuals.shape[1]
        
        if self.use_quantile_regression:
            print("  Using quantile regression for adaptive intervals")
            # For QR, we'll train models during prediction
            # Store residuals for later
            self.calib_residuals = residuals
            self.Y_pred_calib = Y_pred_calib
        else:
            print("  Using fixed quantiles")
            # Compute fixed quantile for each coordinate
            for j in range(n_coords):
                n = len(residuals)
                q = np.ceil((n + 1) * (1 - self.alpha)) / n
                self.quantiles[j] = np.quantile(residuals[:, j], q)
    
    def predict_intervals(self, Y_pred_test, Y_test=None):
        """
        Predict intervals for test data
        
        Args:
            Y_pred_test: Predicted centers (n_test, n_coords)
            Y_test: Optional true values for creating residuals
        """
        n_test, n_coords = Y_pred_test.shape
        print(f"  Predicting intervals for {n_test} samples")
        
        if self.use_quantile_regression and Y_test is not None:
            # Use quantile regression with sequential updates
            print("  Training coordinate-wise quantile forests...")
            self.intervals = np.zeros((n_test, n_coords, 2))
            
            for j in range(n_coords):
                if j % 4 == 0:  # Print progress
                    print(f"    Coordinate {j+1}/{n_coords}")
                
                # Prepare training data: use past residuals as features
                residuals_j = self.calib_residuals[:, j]
                
                # Create simple lagged features (past residual predicts future quantile)
                # Use last 5 residuals as features
                window = min(5, len(residuals_j))
                X_train = []
                y_train = []
                
                for i in range(window, len(residuals_j)):
                    X_train.append(residuals_j[i-window:i])
                    y_train.append(residuals_j[i])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                if len(X_train) > 10:  # Need enough data
                    # Train QRF with the desired quantile
                    # FIXED: Pass quantiles during initialization
                    qrf = RandomForestQuantileRegressor(
                        n_estimators=10,
                        max_depth=2,
                        q=[1 - self.alpha],  # Specify quantiles here!
                        n_jobs=-1
                    )
                    qrf.fit(X_train, y_train)
                    
                    # For each test point, predict the quantile
                    for i in range(n_test):
                        # Use last 'window' calibration residuals as context
                        X_test_i = residuals_j[-window:].reshape(1, -1)
                        
                        # FIXED: Don't pass quantile parameter to predict()
                        # The model already knows which quantiles to predict
                        quantile_pred = qrf.predict(X_test_i)[0]
                        
                        # Create symmetric interval
                        self.intervals[i, j, 0] = Y_pred_test[i, j] - quantile_pred
                        self.intervals[i, j, 1] = Y_pred_test[i, j] + quantile_pred
                        
                        # Update residuals with actual residual if available
                        if Y_test is not None and i < len(Y_test):
                            new_resid = np.abs(Y_test[i, j] - Y_pred_test[i, j])
                            residuals_j = np.append(residuals_j, new_resid)
                else:
                    # Fallback to empirical quantile
                    q_val = np.quantile(residuals_j, 1 - self.alpha)
                    for i in range(n_test):
                        self.intervals[i, j, 0] = Y_pred_test[i, j] - q_val
                        self.intervals[i, j, 1] = Y_pred_test[i, j] + q_val
        else:
            # Use fixed quantiles
            print("  Using fixed quantiles for all test points")
            self.intervals = np.zeros((n_test, n_coords, 2))
            
            for j in range(n_coords):
                q_val = self.quantiles[j]
                self.intervals[:, j, 0] = Y_pred_test[:, j] - q_val
                self.intervals[:, j, 1] = Y_pred_test[:, j] + q_val
        
        return self.intervals
    
    def evaluate_coverage(self, Y_true):
        """
        Evaluate coverage and average volume
        
        Args:
            Y_true: True values (n_test, n_coords)
            
        Returns:
            coverage: Fraction of points inside intervals
            volume: Average interval volume
        """
        if self.intervals is None:
            raise ValueError("Must call predict_intervals first")
        
        # Check if each coordinate is covered
        lower = self.intervals[:, :, 0]
        upper = self.intervals[:, :, 1]
        
        # Coordinate-wise coverage
        coord_covered = (Y_true >= lower) & (Y_true <= upper)
        
        # Joint coverage: all coordinates must be covered
        joint_covered = np.all(coord_covered, axis=1)
        coverage = np.mean(joint_covered)
        
        # Volume: product of interval widths
        widths = upper - lower
        volumes = np.prod(widths, axis=1)
        avg_volume = np.mean(volumes)
        
        return coverage, avg_volume