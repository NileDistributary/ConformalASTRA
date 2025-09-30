"""
FINAL FIX: Use the predictions already stored in SPCI data.
No need to call the model again - we have everything we need!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def extract_ellipsoid_from_existing_data(spci, sample_idx=0, alpha=0.1):
    """
    Extract ellipsoid using the data already computed by SPCI.
    No model prediction needed - use the stored results!
    """
    print(f"\nExtracting ellipsoid from existing SPCI data for sample {sample_idx}...")
    
    try:
        # 1. Get the center from ensemble mean (already computed predictions)
        if hasattr(spci, 'Ensemble_pred_interval_centers') and len(spci.Ensemble_pred_interval_centers) > sample_idx:
            # Use the mean of ensemble predictions as center
            center = np.mean(spci.Ensemble_pred_interval_centers, axis=0)
            print(f"✓ Center from ensemble mean: shape {center.shape}")
        else:
            print("✗ No ensemble predictions found")
            return None, None, None
        
        # 2. Get covariance matrix (already computed by SPCI)
        if hasattr(spci, 'global_cov'):
            cov_matrix = spci.global_cov
            print(f"✓ Using global covariance: shape {cov_matrix.shape}")
        else:
            # Compute from ensemble predictions
            cov_matrix = np.cov(spci.Ensemble_pred_interval_centers.T)
            print(f"✓ Computed covariance from ensemble: shape {cov_matrix.shape}")
        
        # 3. Get radius from SPCI width calculation
        if hasattr(spci, 'Width_Ensemble') and len(spci.Width_Ensemble) > sample_idx:
            # Get the width for this sample
            width_row = spci.Width_Ensemble.iloc[sample_idx]
            if hasattr(width_row, 'values') and len(width_row.values) > 0:
                radius_squared = np.max(width_row.values)
            else:
                radius_squared = float(width_row.iloc[0]) if hasattr(width_row, 'iloc') else float(width_row)
            print(f"✓ Radius from SPCI width: {radius_squared}")
        else:
            # Fallback: use chi-squared quantile
            radius_squared = chi2.ppf(1 - alpha, df=len(center))
            print(f"⚠ Using chi2 radius: {radius_squared}")
        
        # 4. Validate dimensions
        if len(center) != 24:
            print(f"⚠ Unexpected center dimension: {len(center)}, expected 24")
        
        if cov_matrix.shape != (24, 24):
            print(f"⚠ Unexpected covariance shape: {cov_matrix.shape}, expected (24, 24)")
        
        print(f"✓ Successfully extracted ellipsoid parameters")
        return center, cov_matrix, radius_squared
        
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def plot_ellipses_from_spci_data(spci, sample_indices=[0, 1, 2], confidence_level=0.9):
    """
    Plot ellipses using SPCI data without calling the model.
    """
    print("\n" + "="*60)
    print("PLOTTING ELLIPSES FROM EXISTING SPCI DATA")
    print("="*60)
    
    # Extract ellipsoid parameters once (same for all samples since we use ensemble mean)
    center, cov_matrix, radius_squared = extract_ellipsoid_from_existing_data(
        spci, sample_idx=0, alpha=1-confidence_level
    )
    
    if center is None:
        print("Cannot plot - extraction failed")
        return
    
    # Reshape center to trajectory format
    center_traj = center.reshape(12, 2)  # 12 timesteps × 2 coordinates
    
    # Create plots
    n_samples = len(sample_indices)
    timesteps_to_show = [0, 2, 5, 8, 11]  # Show key timesteps
    
    fig, axes = plt.subplots(2, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 8))
    
    # Row 1: Individual timestep ellipses
    for j, timestep in enumerate(timesteps_to_show):
        ax = axes[0, j]
        
        # Extract 2D marginal for this timestep
        start_idx = timestep * 2
        end_idx = start_idx + 2
        
        center_2d = center[start_idx:end_idx]
        cov_2d = cov_matrix[start_idx:end_idx, start_idx:end_idx]
        
        print(f"Timestep {timestep+1}: center={center_2d}, cov_det={np.linalg.det(cov_2d):.6f}")
        
        # Plot ellipse
        plot_2d_ellipse_robust(ax, center_2d, cov_2d, confidence_level, 'blue', alpha=0.4)
        
        ax.set_title(f'Timestep {timestep+1}\n({confidence_level*100:.0f}% Confidence)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Set reasonable limits
        margin = 3 * np.sqrt(np.max(np.diag(cov_2d)))
        ax.set_xlim(center_2d[0] - margin, center_2d[0] + margin)
        ax.set_ylim(center_2d[1] - margin, center_2d[1] + margin)
    
    # Row 2: Complete trajectory with all ellipses
    ax = axes[1, 2]  # Use middle subplot for trajectory
    
    # Hide other subplots in row 2
    for j in range(len(timesteps_to_show)):
        if j != 2:
            axes[1, j].set_visible(False)
    
    # Colors for time progression
    colors = plt.cm.viridis(np.linspace(0, 1, 12))
    
    # Plot ellipse for each timestep
    for timestep in range(12):
        start_idx = timestep * 2
        end_idx = start_idx + 2
        
        center_2d = center[start_idx:end_idx]
        cov_2d = cov_matrix[start_idx:end_idx, start_idx:end_idx]
        
        # Plot smaller ellipses
        plot_2d_ellipse_robust(ax, center_2d, cov_2d, confidence_level, colors[timestep], alpha=0.3)
    
    # Plot trajectory line
    ax.plot(center_traj[:, 0], center_traj[:, 1], 'k-', linewidth=3, alpha=0.8, label='Predicted trajectory')
    
    # Mark start and end
    ax.scatter(center_traj[0, 0], center_traj[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(center_traj[-1, 0], center_traj[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    
    # Add timestep annotations
    for t in range(0, 12, 2):  # Every other timestep
        ax.annotate(f'{t+1}', center_traj[t], xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_title(f'Complete Trajectory with Uncertainty Ellipses\n({confidence_level*100:.0f}% Confidence)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=12))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Timestep')
    
    plt.tight_layout()
    plt.show()


def plot_2d_ellipse_robust(ax, center, cov_matrix, confidence_level, color, alpha=0.5, linewidth=2):
    """
    Robust 2D ellipse plotting with error handling.
    """
    try:
        # Ensure covariance is positive definite
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Fix negative or zero eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-8)
        
        # Reconstruct covariance if needed
        if np.min(eigenvals) < 1e-6:
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Chi-squared critical value for 2D
        chi2_val = chi2.ppf(confidence_level, df=2)
        
        # Ellipse parameters
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])
        
        # Create ellipse
        ellipse = Ellipse(center, width, height, angle=angle, 
                         facecolor=color, edgecolor='black', 
                         alpha=alpha, linewidth=linewidth)
        ax.add_patch(ellipse)
        
        # Mark center
        ax.scatter(center[0], center[1], c='black', s=20, zorder=5)
        
        print(f"  ✓ Plotted ellipse: center={center}, width={width:.3f}, height={height:.3f}, angle={angle:.1f}°")
        
    except Exception as e:
        print(f"  ✗ Failed to plot ellipse: {e}")
        # Fallback: plot a circle
        circle = plt.Circle(center, 0.1, facecolor=color, edgecolor='black', alpha=alpha)
        ax.add_patch(circle)
        ax.scatter(center[0], center[1], c='black', s=20, zorder=5)


def plot_simple_comparison_ellipses(spci):
    """
    Simple plot to compare different samples using ensemble data.
    """
    print("\n" + "="*60)
    print("COMPARING ELLIPSES FOR DIFFERENT SAMPLES")
    print("="*60)
    
    # Get ensemble predictions for comparison
    if not hasattr(spci, 'Ensemble_pred_interval_centers'):
        print("No ensemble data available")
        return
    
    ensemble_preds = spci.Ensemble_pred_interval_centers
    n_samples = min(3, len(ensemble_preds))
    
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Use this sample's prediction as center
        center = ensemble_preds[i]
        center_traj = center.reshape(12, 2)
        
        # Use global covariance
        if hasattr(spci, 'global_cov'):
            cov_matrix = spci.global_cov
        else:
            cov_matrix = np.cov(ensemble_preds.T)
        
        # Plot trajectory with ellipses at key points
        timesteps = [0, 5, 11]  # Start, middle, end
        colors = ['green', 'blue', 'red']
        
        for j, timestep in enumerate(timesteps):
            start_idx = timestep * 2
            end_idx = start_idx + 2
            
            center_2d = center[start_idx:end_idx]
            cov_2d = cov_matrix[start_idx:end_idx, start_idx:end_idx]
            
            plot_2d_ellipse_robust(ax, center_2d, cov_2d, 0.9, colors[j], alpha=0.3)
        
        # Plot trajectory
        ax.plot(center_traj[:, 0], center_traj[:, 1], 'k-', linewidth=2, alpha=0.7)
        
        ax.set_title(f'Sample {i}\nEnsemble Prediction')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.show()


def debug_spci_data(spci):
    """
    Debug function to understand the SPCI data structure.
    """
    print("\n" + "="*50)
    print("DEBUGGING SPCI DATA FOR VISUALIZATION")
    print("="*50)
    
    # Check ensemble predictions
    if hasattr(spci, 'Ensemble_pred_interval_centers'):
        ensemble = spci.Ensemble_pred_interval_centers
        print(f"✓ Ensemble predictions: shape {ensemble.shape}")
        print(f"  Sample mean: {np.mean(ensemble, axis=0)[:4]}...")
        print(f"  Sample std: {np.std(ensemble, axis=0)[:4]}...")
    
    # Check covariance
    if hasattr(spci, 'global_cov'):
        cov = spci.global_cov
        print(f"✓ Global covariance: shape {cov.shape}")
        print(f"  Diagonal (first 4): {np.diag(cov)[:4]}")
        print(f"  Determinant: {np.linalg.det(cov):.2e}")
    
    # Check widths
    if hasattr(spci, 'Width_Ensemble'):
        widths = spci.Width_Ensemble
        print(f"✓ Width data: shape {widths.shape}")
        print(f"  First few widths: {widths.iloc[:3]}")
    
    print("="*50)


# Main function to use
def visualize_spci_uncertainty_ellipses(spci):
    """
    Main function to visualize uncertainty ellipses from SPCI data.
    This bypasses the model prediction issue completely.
    
    Usage:
    ------
    visualize_spci_uncertainty_ellipses(spci)
    """
    
    print("Visualizing uncertainty ellipses from SPCI data...")
    
    # Debug first
    debug_spci_data(spci)
    
    # Main visualization
    plot_ellipses_from_spci_data(spci, sample_indices=[0, 1, 2], confidence_level=0.9)
    
    # Comparison plot
    plot_simple_comparison_ellipses(spci)
    
    print("\n" + "="*60)
    print("✓ VISUALIZATION COMPLETE!")
    print("You should now see actual elliptical uncertainty regions!")
    print("="*60)


if __name__ == "__main__":
    print("Use this function with your SPCI results:")
    print("visualize_spci_uncertainty_ellipses(spci)")