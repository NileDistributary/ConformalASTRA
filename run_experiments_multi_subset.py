"""
Multi-Subset Experiment Pipeline
Runs specified experiments across multiple ETH-UCY subsets with proper organization.

This script provides a flexible framework to:
1. Run the same experiment across multiple subsets
2. Organize results by subset
3. Generate comparative analysis across subsets
4. Handle subset-specific configurations automatically
"""

import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import os
from pathlib import Path
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

# Available subsets with their expected pretrained model names
AVAILABLE_SUBSETS = {
    'eth': {
        'config': 'configs/eth.yaml',
        'model': 'pretrained_astra_weights/eth_best_model.pth',
        'description': 'ETH Pedestrian Dataset'
    },
    'hotel': {
        'config': 'configs/hotel.yaml',
        'model': 'pretrained_astra_weights/hotel_best_model.pth',
        'description': 'Hotel Scene'
    },
    'univ': {
        'config': 'configs/univ.yaml',
        'model': 'pretrained_astra_weights/univ_best_model.pth',
        'description': 'University Scene'
    },
    'zara01': {
        'config': 'configs/zara01.yaml',
        'model': 'pretrained_astra_weights/zara01_best_model.pth',
        'description': 'Zara 01 Scene'
    },
    'zara02': {
        'config': 'configs/zara02.yaml',
        'model': 'pretrained_astra_weights/zara02_best_model.pth',
        'description': 'Zara 02 Scene'
    }
}

# Available experiments
AVAILABLE_EXPERIMENTS = {
    'rank': {
        'script': 'experiment_scripts/experiment_ablation_rank.py',
        'description': 'Covariance rank approximation ablation'
    },
    'calibration_size': {
        'script': 'experiment_scripts/experiment_ablation_calibration_size.py',
        'description': 'Calibration set size ablation'
    },
    'qr_vs_empirical': {
        'script': 'experiment_scripts/experiment_ablation_qr_vs_empirical.py',
        'description': 'Quantile regression vs empirical quantile'
    },
    'past_window': {
        'script': 'experiment_scripts/experiment_ablation_past_window.py',
        'description': 'Past window size ablation'
    },
    'global_vs_local': {
        'script': 'experiment_scripts/experiment_ablation_global_vs_local.py',
        'description': 'Global vs local covariance comparison'
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(text, char="=", width=80):
    """Print formatted header"""
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width + "\n")


def check_subset_availability(subset):
    """
    Check if a subset has all required files.
    
    Returns:
        tuple: (bool, list of missing files)
    """
    if subset not in AVAILABLE_SUBSETS:
        return False, [f"Unknown subset: {subset}"]
    
    subset_info = AVAILABLE_SUBSETS[subset]
    missing = []
    
    # Check config file
    if not os.path.exists(subset_info['config']):
        missing.append(f"Config: {subset_info['config']}")
    
    # Check model weights
    if not os.path.exists(subset_info['model']):
        missing.append(f"Model: {subset_info['model']}")
    
    return len(missing) == 0, missing


def check_experiment_availability(experiment):
    """Check if an experiment script exists"""
    if experiment not in AVAILABLE_EXPERIMENTS:
        return False, f"Unknown experiment: {experiment}"
    
    script_path = AVAILABLE_EXPERIMENTS[experiment]['script']
    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"
    
    return True, None


def modify_experiment_for_subset(script_path, subset, temp_dir='temp_subset_scripts'):
    """
    Create a modified version of the experiment script for a specific subset.
    
    This function creates a temporary copy of the experiment script with
    the subset parameter modified to use the specified subset.
    
    Args:
        script_path: Path to the original experiment script
        subset: The subset to use (e.g., 'hotel', 'eth')
        temp_dir: Directory to store temporary modified scripts
        
    Returns:
        Path to the modified script
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # Read original script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Modify the subset parameter
    # Look for lines like: subset = 'eth'
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        # Match patterns like: subset = 'eth' or config_path = 'configs/eth.yaml'
        if "subset = " in line and "'" in line:
            # Extract indentation
            indent = line[:len(line) - len(line.lstrip())]
            modified_lines.append(f"{indent}subset = '{subset}'")
        elif "config_path = " in line and "configs/" in line:
            indent = line[:len(line) - len(line.lstrip())]
            modified_lines.append(f"{indent}config_path = 'configs/{subset}.yaml'")
        else:
            modified_lines.append(line)
    
    modified_content = '\n'.join(modified_lines)
    
    # Create temporary script
    script_name = os.path.basename(script_path)
    temp_script_path = os.path.join(temp_dir, f"{subset}_{script_name}")
    
    with open(temp_script_path, 'w') as f:
        f.write(modified_content)
    
    return temp_script_path


def run_experiment_on_subset(experiment, subset, dry_run=False):
    """
    Run a single experiment on a single subset.
    
    Returns:
        dict: Result summary with keys: subset, experiment, status, time, error
    """
    print_header(f"Running {experiment} on {subset}", char="-", width=60)
    
    # Verify availability
    subset_ok, subset_missing = check_subset_availability(subset)
    if not subset_ok:
        print(f"✗ Subset '{subset}' not ready:")
        for item in subset_missing:
            print(f"  - Missing: {item}")
        return {
            'subset': subset,
            'experiment': experiment,
            'status': 'subset_not_ready',
            'time': 0,
            'error': '; '.join(subset_missing)
        }
    
    exp_ok, exp_error = check_experiment_availability(experiment)
    if not exp_ok:
        print(f"✗ Experiment '{experiment}' not available: {exp_error}")
        return {
            'subset': subset,
            'experiment': experiment,
            'status': 'experiment_not_found',
            'time': 0,
            'error': exp_error
        }
    
    print(f"✓ All prerequisites met")
    print(f"Config: {AVAILABLE_SUBSETS[subset]['config']}")
    print(f"Model: {AVAILABLE_SUBSETS[subset]['model']}")
    print(f"Script: {AVAILABLE_EXPERIMENTS[experiment]['script']}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if dry_run:
        print("\n[DRY RUN] Would execute experiment here\n")
        return {
            'subset': subset,
            'experiment': experiment,
            'status': 'dry_run',
            'time': 0,
            'error': None
        }
    
    # Create modified script for this subset
    script_path = AVAILABLE_EXPERIMENTS[experiment]['script']
    temp_script = modify_experiment_for_subset(script_path, subset)
    
    print(f"\nExecuting modified script: {temp_script}")
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            check=False
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Check status
        if result.returncode == 0:
            print(f"\n✓ COMPLETED")
            print(f"Time: {elapsed:.1f}s")
            status = 'success'
            error = None
        else:
            print(f"\n✗ FAILED (return code: {result.returncode})")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
            status = 'failed'
            error = result.stderr[:200] if result.stderr else "Unknown error"
        
        return {
            'subset': subset,
            'experiment': experiment,
            'status': status,
            'time': elapsed,
            'error': error
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ EXCEPTION: {str(e)}")
        return {
            'subset': subset,
            'experiment': experiment,
            'status': 'exception',
            'time': elapsed,
            'error': str(e)
        }
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)


def generate_summary_report(results, output_dir='results/multi_subset_summary'):
    """Generate and save a summary report of all experiments"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save full results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f'experiment_summary_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to: {csv_path}")
    
    # Print summary table
    print_header("EXPERIMENT SUMMARY", width=80)
    
    # Group by status
    print("\nStatus Summary:")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Results by subset
    print("\nResults by Subset:")
    for subset in df['subset'].unique():
        subset_df = df[df['subset'] == subset]
        success = len(subset_df[subset_df['status'] == 'success'])
        total = len(subset_df)
        print(f"  {subset}: {success}/{total} successful")
    
    # Results by experiment
    print("\nResults by Experiment:")
    for exp in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp]
        success = len(exp_df[exp_df['status'] == 'success'])
        total = len(exp_df)
        print(f"  {exp}: {success}/{total} successful")
    
    # Failed experiments
    failed = df[df['status'] != 'success']
    if len(failed) > 0:
        print("\nFailed Experiments:")
        for _, row in failed.iterrows():
            print(f"  {row['subset']} - {row['experiment']}: {row['error']}")
    
    print("=" * 80)
    
    return csv_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run experiments across multiple ETH-UCY subsets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run rank ablation on hotel and eth subsets
  python run_experiments_multi_subset.py --experiments rank --subsets hotel eth
  
  # Run all experiments on all available subsets
  python run_experiments_multi_subset.py --experiments all --subsets all
  
  # Dry run to check what would be executed
  python run_experiments_multi_subset.py --experiments rank --subsets hotel --dry-run
  
  # Check which subsets are ready
  python run_experiments_multi_subset.py --check-subsets
        """
    )
    
    parser.add_argument('--experiments', nargs='+', 
                       help='Experiments to run (use "all" for all experiments)')
    parser.add_argument('--subsets', nargs='+',
                       help='Subsets to use (use "all" for all subsets)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Check configuration without running experiments')
    parser.add_argument('--check-subsets', action='store_true',
                       help='Check which subsets are ready and exit')
    parser.add_argument('--list-experiments', action='store_true',
                       help='List available experiments and exit')
    
    args = parser.parse_args()
    
    # Handle info requests
    if args.check_subsets:
        print_header("SUBSET AVAILABILITY CHECK")
        for subset, info in AVAILABLE_SUBSETS.items():
            available, missing = check_subset_availability(subset)
            status = "✓ READY" if available else "✗ NOT READY"
            print(f"\n{subset} ({info['description']}): {status}")
            if not available:
                for item in missing:
                    print(f"  Missing: {item}")
        return
    
    if args.list_experiments:
        print_header("AVAILABLE EXPERIMENTS")
        for exp, info in AVAILABLE_EXPERIMENTS.items():
            print(f"\n{exp}:")
            print(f"  Description: {info['description']}")
            print(f"  Script: {info['script']}")
            exists = "✓" if os.path.exists(info['script']) else "✗"
            print(f"  Status: {exists}")
        return
    
    # Validate arguments
    if not args.experiments or not args.subsets:
        parser.print_help()
        return
    
    # Parse experiments
    if 'all' in args.experiments:
        experiments = list(AVAILABLE_EXPERIMENTS.keys())
    else:
        experiments = args.experiments
        # Validate
        for exp in experiments:
            if exp not in AVAILABLE_EXPERIMENTS:
                print(f"Error: Unknown experiment '{exp}'")
                print(f"Available: {list(AVAILABLE_EXPERIMENTS.keys())}")
                return
    
    # Parse subsets
    if 'all' in args.subsets:
        subsets = list(AVAILABLE_SUBSETS.keys())
    else:
        subsets = args.subsets
        # Validate
        for subset in subsets:
            if subset not in AVAILABLE_SUBSETS:
                print(f"Error: Unknown subset '{subset}'")
                print(f"Available: {list(AVAILABLE_SUBSETS.keys())}")
                return
    
    # Print execution plan
    print_header("EXECUTION PLAN")
    print(f"Subsets to process: {', '.join(subsets)}")
    print(f"Experiments to run: {', '.join(experiments)}")
    print(f"Total combinations: {len(subsets) * len(experiments)}")
    print(f"Dry run: {args.dry_run}")
    
    # Confirm with user unless dry run
    if not args.dry_run:
        response = input("\nProceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Execute experiments
    all_results = []
    total = len(subsets) * len(experiments)
    current = 0
    
    print_header(f"STARTING EXPERIMENTS", width=80)
    start_time = time.time()
    
    for subset in subsets:
        for experiment in experiments:
            current += 1
            print(f"\n[{current}/{total}] {subset} - {experiment}")
            
            result = run_experiment_on_subset(
                experiment=experiment,
                subset=subset,
                dry_run=args.dry_run
            )
            all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary
    if not args.dry_run:
        summary_path = generate_summary_report(all_results)
    
    print(f"\nTotal execution time: {total_time:.1f}s")
    print(f"Average per experiment: {total_time/total:.1f}s")
    
    print_header("ALL EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
