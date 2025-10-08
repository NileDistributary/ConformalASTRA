"""
Master Script: Run All Ablation Studies
Executes all five ablation studies sequentially and generates a summary report.
"""
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_experiment(script_name, description):
    """Run a single experiment script with robust error handling"""
    print_header(f"STARTING: {description}")
    print(f"Running: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if script exists
    if not os.path.exists(script_name):
        print(f"✗ ERROR: Script not found: {script_name}")
        return {
            'experiment': description,
            'status': 'script_not_found',
            'time': 0,
            'error': f"Script {script_name} does not exist"
        }
    
    start_time = time.time()
    
    try:
        # Run the script (check=False to not raise exception on non-zero exit)
        # NO TIMEOUT - experiments can take as long as needed
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception, we'll handle it manually
        )
        
        elapsed = time.time() - start_time
        
        # Print stdout (the experiment's output)
        if result.stdout:
            print(result.stdout)
        
        # Check if the script succeeded
        if result.returncode == 0:
            print(f"\n✓ COMPLETED: {description}")
            print(f"Time taken: {elapsed:.1f} seconds")
            
            return {
                'experiment': description,
                'status': 'success',
                'time': elapsed,
                'error': None
            }
        else:
            # Script failed but we can continue
            print(f"\n✗ FAILED: {description}")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            
            return {
                'experiment': description,
                'status': 'failed',
                'time': elapsed,
                'error': f"Exit code {result.returncode}: {result.stderr[:500]}"  # Truncate long errors
            }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ UNEXPECTED ERROR: {description}")
        print(f"Exception: {type(e).__name__}: {str(e)}")
        
        return {
            'experiment': description,
            'status': 'error',
            'time': elapsed,
            'error': f"{type(e).__name__}: {str(e)}"
        }


def generate_summary_report(results):
    """Generate a summary report of all experiments"""
    print_header("ABLATION STUDIES SUMMARY REPORT")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print detailed table
    print("Experiment Execution Summary:")
    print("-" * 100)
    print(f"{'Status':<8} {'Experiment':<55} {'Time':<12} {'Status Details'}")
    print("-" * 100)
    
    status_symbols = {
        'success': '✓',
        'failed': '✗',
        'error': '⚠',
        'script_not_found': '?'
    }
    
    for i, row in df.iterrows():
        symbol = status_symbols.get(row['status'], '?')
        time_str = f"{row['time']:.1f}s"
        error_preview = ""
        if row['error']:
            # Show first 30 chars of error
            error_preview = str(row['error'])[:30] + "..." if len(str(row['error'])) > 30 else str(row['error'])
        
        print(f"{symbol:<8} {row['experiment']:<55} {time_str:<12} {error_preview}")
    print("-" * 100)
    
    # Statistics
    total_time = df['time'].sum()
    success_count = (df['status'] == 'success').sum()
    failed_count = (df['status'] == 'failed').sum()
    error_count = (df['status'] == 'error').sum()
    not_found_count = (df['status'] == 'script_not_found').sum()
    total_count = len(df)
    
    print(f"\n Statistics:")
    print(f"  Total experiments: {total_count}")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    if error_count > 0:
        print(f"  ⚠ Errors: {error_count}")
    if not_found_count > 0:
        print(f"  ? Not found: {not_found_count}")
    print(f"  ⏱ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Success rate
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"   Success rate: {success_rate:.1f}%")
    
    # Save summary to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'./results/csvs/ablation_summary_{timestamp}.csv'
    
    os.makedirs('./results/csvs', exist_ok=True)
    df.to_csv(summary_file, index=False)
    print(f"\n Summary saved to: {summary_file}")
    
    # If there were failures, save detailed error log
    if success_count < total_count:
        error_file = f'./results/errors/ablation_errors_{timestamp}.txt'
        os.makedirs('./results/errors', exist_ok=True)
        with open(error_file, 'w') as f:
            f.write("ABLATION STUDIES ERROR LOG\n")
            f.write("="*80 + "\n\n")
            for i, row in df.iterrows():
                if row['status'] != 'success':
                    f.write(f"Experiment: {row['experiment']}\n")
                    f.write(f"Status: {row['status']}\n")
                    f.write(f"Time: {row['time']:.1f}s\n")
                    f.write(f"Error: {row['error']}\n")
                    f.write("-"*80 + "\n\n")
        print(f"⚠ Error details saved to: {error_file}")
    
    return df


def main():
    """Main execution function"""
    print_header("CONFORMAL ASTRA - COMPLETE ABLATION STUDY SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all experiments (now in experiment_scripts folder)
    experiments = [
        {
            'script': 'experiment_scripts/experiment_baseline_comparison.py',
            'description': 'Baseline Comparison (Coordinate-wise vs MultiDimSPCI)'
        },
        {
            'script': 'experiment_scripts/experiment_ablation_rank.py',
            'description': 'Ablation 1: Rank Approximation'
        },
        {
            'script': 'experiment_scripts/experiment_ablation_calibration_size.py',
            'description': 'Ablation 2: Calibration Set Size'
        },
        {
            'script': 'experiment_scripts/experiment_ablation_qr_vs_empirical.py',
            'description': 'Ablation 3: Quantile Regression vs Empirical Quantile'
        },
        {
            'script': 'experiment_scripts/experiment_ablation_past_window.py',
            'description': 'Ablation 4: Past Window Size'
        },
        {
            'script': 'experiment_scripts/experiment_ablation_global_vs_local.py',
            'description': 'Ablation 5: Global vs Local Ellipsoids'
        }
    ]
    
    total_experiments = len(experiments)
    print(f"Total experiments to run: {total_experiments}\n")
    
    # Run all experiments (continue even if some fail)
    results = []
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {idx}/{total_experiments}")
        print(f"{'#'*80}\n")
        
        result = run_experiment(exp['script'], exp['description'])
        results.append(result)
        
        # Print progress
        completed = idx
        success_so_far = sum(1 for r in results if r['status'] == 'success')
        print(f"\nProgress: {completed}/{total_experiments} completed, {success_so_far} successful")
        
        # Small pause between experiments to avoid resource conflicts
        if idx < total_experiments:
            print("\nWaiting 3 seconds before next experiment...")
            time.sleep(3)
    
    # Generate summary
    print("\n" + "="*80)
    df = generate_summary_report(results)
    
    print_header("ALL ABLATION STUDIES COMPLETED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check results and provide recommendations
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print(f"\n⚠ WARNING: {len(failed)}/{total_experiments} experiments did not complete successfully.")
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['experiment']} (status: {r['status']})")
        print("\nCheck the error log in results/errors/ for details.")
        print("You can re-run individual experiments manually to debug issues.")
        return 1
    else:
        print("\n✓✓✓ SUCCESS! All experiments completed successfully! ✓✓✓")
        print("\nResults saved to:")
        print("  - CSV files: results/csvs/")
        print("  - Figures: results/figures/")
        print("  - Configs: results/configs/")
        print("\nNext steps:")
        print("  1. Analyze the CSV files with metrics")
        print("  2. Review the visualizations")
        print("  3. Draw conclusions for your paper")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)