"""
Coordinate-wise Conformal Prediction Baseline

Applies 1D conformal prediction independently to each coordinate (x, y at each timestep).
This is the naive baseline that MultiDimSPCI should outperform.
"""
import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor


class CoordinateWiseCP:
    """
    Coordinate-wise conformal prediction.
    
    For a 24D trajectory (12 timesteps × 2 coords), this creates 24 separate
    1D prediction intervals, resulting in a hyper-rectangular region.
    """
    
    def __init__(self, alpha=0.1, use_quantile_regression=True, past_window=100):
        """
        Parameters:
        -----------
        alpha : float
            Significance level (1-alpha = coverage target)
        use_quantile_regression : bool
            If True, use quantile regression. If False, use empirical quantile.
        past_window : int
            Window size for quantile regression
        """
        self.alpha = alpha
        self.use_quantile_regression = use_quantile_regression
        self.past_window = past_window
        
        self.calib_residuals = None
        self.test_residuals = None
        self.intervals = None
        
    def calibrate(self, Y_calib, Y_pred_calib):
        """
        Compute calibration residuals for each coordinate
        
        Parameters:
        -----------
        Y_calib : np.array, shape (n_calib, 24)
            True values on calibration set
        Y_pred_calib : np.array, shape (n_calib, 24)
            Predicted values on calibration set
        """
        # Compute absolute residuals for each coordinate
        self.calib_residuals = np.abs(Y_calib - Y_pred_calib)  # Shape: (n_calib, 24)
        
        n_coords = self.calib_residuals.shape[1]
        print(f"✓ Calibration: {self.calib_residuals.shape[0]} samples, {n_coords} coordinates")
        
    def predict_intervals(self, Y_pred_test, Y_test=None):
        """
        Compute prediction intervals for test set
        
        Parameters:
        -----------
        Y_pred_test : np.array, shape (n_test, 24)
            Predicted values on test set
        Y_test : np.array, shape (n_test, 24), optional
            True values for coverage evaluation
            
        Returns:
        --------
        intervals : dict with keys 'lower', 'upper', each shape (n_test, 24)
        """
        n_test, n_coords = Y_pred_test.shape
        
        if self.use_quantile_regression:
            # Use quantile regression for each coordinate
            intervals_lower = np.zeros((n_test, n_coords))
            intervals_upper = np.zeros((n_test, n_coords))
            
            for coord_idx in range(n_coords):
                # Get residuals for this coordinate
                coord_residuals = self.calib_residuals[:, coord_idx]
                
                if len(coord_residuals) < self.past_window:
                    # Not enough data, use empirical quantile
                    quantile = np.quantile(coord_residuals, 1 - self.alpha)
                    intervals_lower[:, coord_idx] = Y_pred_test[:, coord_idx] - quantile
                    intervals_upper[:, coord_idx] = Y_pred_test[:, coord_idx] + quantile
                else:
                    # Use quantile regression
                    # Create sliding windows
                    X_train = []
                    for i in range(len(coord_residuals) - self.past_window):
                        X_train.append(coord_residuals[i:i+self.past_window])
                    X_train = np.array(X_train)
                    y_train = coord_residuals[self.past_window:]
                    
                    # Fit quantile regressor
                    qrf = RandomForestQuantileRegressor(
                        n_estimators=10,
                        max_depth=2,
                        random_state=42
                    )
                    qrf.fit(X_train, y_train)
                    
                    # Predict quantiles for test set
                    # Use most recent calibration residuals as features
                    for test_idx in range(n_test):
                        if test_idx == 0:
                            X_test = coord_residuals[-self.past_window:].reshape(1, -1)
                        else:
                            # Use previous test residuals if available
                            recent_residuals = coord_residuals[-self.past_window+test_idx:].tolist()
                            if Y_test is not None and test_idx > 0:
                                test_residuals = np.abs(Y_test[:test_idx, coord_idx] - 
                                                       Y_pred_test[:test_idx, coord_idx])
                                recent_residuals.extend(test_residuals.tolist())
                            X_test = np.array(recent_residuals[-self.past_window:]).reshape(1, -1)
                        
                        quantile = qrf.predict(X_test, quantile=1-self.alpha)[0]
                        intervals_lower[test_idx, coord_idx] = Y_pred_test[test_idx, coord_idx] - quantile
                        intervals_upper[test_idx, coord_idx] = Y_pred_test[test_idx, coord_idx] + quantile
        else:
            # Simple empirical quantile for each coordinate
            intervals_lower = np.zeros((n_test, n_coords))
            intervals_upper = np.zeros((n_test, n_coords))
            
            for coord_idx in range(n_coords):
                quantile = np.quantile(self.calib_residuals[:, coord_idx], 1 - self.alpha)
                intervals_lower[:, coord_idx] = Y_pred_test[:, coord_idx] - quantile
                intervals_upper[:, coord_idx] = Y_pred_test[:, coord_idx] + quantile
        
        self.intervals = {
            'lower': intervals_lower,
            'upper': intervals_upper
        }
        
        return self.intervals
    
    def evaluate_coverage(self, Y_test):
        """
        Evaluate empirical coverage on test set
        
        Parameters:
        -----------
        Y_test : np.array, shape (n_test, 24)
            True test values
            
        Returns:
        --------
        coverage : float
            Proportion of test points covered
        avg_volume : float
            Average hyper-rectangle volume
        """
        if self.intervals is None:
            raise ValueError("Must call predict_intervals first")
        
        lower = self.intervals['lower']
        upper = self.intervals['upper']
        
        # Check if each coordinate is covered
        covered = (Y_test >= lower) & (Y_test <= upper)
        
        # A sample is covered if ALL coordinates are covered
        all_coords_covered = np.all(covered, axis=1)
        coverage = np.mean(all_coords_covered)
        
        # Compute average hyper-rectangle volume
        # Volume = product of interval widths
        widths = upper - lower
        volumes = np.prod(widths, axis=1)
        avg_volume = np.mean(volumes)
        
        return coverage, avg_volume
    
    def get_results(self, Y_test):
        """Convenience method matching SPCI interface"""
        return self.evaluate_coverage(Y_test)