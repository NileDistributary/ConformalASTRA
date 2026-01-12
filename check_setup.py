#!/usr/bin/env python3
"""
Multi-Subset Setup Diagnostic Tool

Run this script to verify your ConformalASTRA repository is ready for
multi-subset experiments.

Usage:
    python check_setup.py
    python check_setup.py --verbose
    python check_setup.py --subset hotel
"""

import os
import sys
from pathlib import Path
import argparse


# ANSI color codes for pretty output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text, color=Colors.BLUE):
    """Print a colored header"""
    print(f"\n{color}{Colors.BOLD}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.ENDC}\n")


def check_file(filepath, description=""):
    """Check if a file exists and return status"""
    exists = os.path.exists(filepath)
    status = f"{Colors.GREEN}✓{Colors.ENDC}" if exists else f"{Colors.RED}✗{Colors.ENDC}"
    
    if description:
        print(f"  {status} {description}")
        if not exists:
            print(f"      {Colors.YELLOW}Missing: {filepath}{Colors.ENDC}")
    
    return exists


def check_directory(dirpath, description=""):
    """Check if a directory exists and list contents if verbose"""
    exists = os.path.exists(dirpath)
    status = f"{Colors.GREEN}✓{Colors.ENDC}" if exists else f"{Colors.RED}✗{Colors.ENDC}"
    
    print(f"  {status} {description}")
    if not exists:
        print(f"      {Colors.YELLOW}Missing: {dirpath}{Colors.ENDC}")
    
    return exists


def check_subset(subset_name, verbose=False):
    """
    Check if all files for a subset are present.
    
    Returns:
        tuple: (all_present, missing_files)
    """
    print(f"\n{Colors.BOLD}{subset_name.upper()}{Colors.ENDC}")
    
    config_file = f"configs/{subset_name}.yaml"
    model_file = f"pretrained_astra_weights/{subset_name}_best_model.pth"
    unet_file = "pretrained_unet_weights/eth_unet_model_best.pt"
    
    missing = []
    
    # Check config
    if not check_file(config_file, f"Config file: {config_file}"):
        missing.append(config_file)
    
    # Check model weights
    if not check_file(model_file, f"Model weights: {model_file}"):
        missing.append(model_file)
    
    # Check U-Net weights (only report once across all subsets)
    if subset_name == 'eth' or verbose:
        if not check_file(unet_file, f"U-Net weights: {unet_file}"):
            if unet_file not in missing:
                missing.append(unet_file)
    
    all_present = len(missing) == 0
    
    if all_present:
        print(f"  {Colors.GREEN}Status: Ready for experiments{Colors.ENDC}")
    else:
        print(f"  {Colors.RED}Status: Not ready - missing {len(missing)} file(s){Colors.ENDC}")
    
    if verbose and not all_present:
        print(f"\n  {Colors.YELLOW}To fix:{Colors.ENDC}")
        for f in missing:
            if 'config' in f:
                print(f"    - Create config file: {f}")
            elif 'pretrained_astra' in f:
                print(f"    - Train or obtain model weights: {f}")
            elif 'unet' in f:
                print(f"    - Download U-Net weights: {f}")
    
    return all_present, missing


def check_experiment_scripts(verbose=False):
    """Check if experiment scripts exist"""
    print_header("EXPERIMENT SCRIPTS", Colors.BLUE)
    
    experiments = {
        'rank': 'experiment_scripts/experiment_ablation_rank.py',
        'calibration': 'experiment_scripts/experiment_ablation_calibration_size.py',
        'qr': 'experiment_scripts/experiment_ablation_qr_vs_empirical.py',
        'window': 'experiment_scripts/experiment_ablation_past_window.py',
        'global_local': 'experiment_scripts/experiment_ablation_global_vs_local.py',
    }
    
    all_present = True
    for name, path in experiments.items():
        exists = check_file(path, f"{name.ljust(12)}: {path}")
        all_present = all_present and exists
    
    return all_present


def check_utilities(verbose=False):
    """Check if utility files and modules exist"""
    print_header("UTILITY FILES", Colors.BLUE)
    
    utilities = {
        'subset_utils': 'subset_utils.py',
        'misc': 'utils/misc.py',
        'astra_wrapper': 'astra_wrapper.py',
        'MultiDimSPCI': 'helpers/MultiDim_SPCI_class.py',
    }
    
    all_present = True
    for name, path in utilities.items():
        # Try both root and utils directories
        paths_to_try = [path, f'utils/{path}']
        exists = False
        for p in paths_to_try:
            if os.path.exists(p):
                exists = True
                check_file(p, f"{name.ljust(15)}: {p}")
                break
        
        if not exists:
            check_file(path, f"{name.ljust(15)}: {path}")
        
        all_present = all_present and exists
    
    return all_present


def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("PYTHON DEPENDENCIES", Colors.BLUE)
    
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'yaml',
        'munch',
        'albumentations',
        'cv2',  # opencv
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            # Special case for cv2
            if package == 'cv2':
                __import__(package)
                name = 'opencv-python'
            else:
                __import__(package)
                name = package
            
            print(f"  {Colors.GREEN}✓{Colors.ENDC} {name}")
        except ImportError:
            print(f"  {Colors.RED}✗{Colors.ENDC} {name} - Not installed")
            all_installed = False
    
    return all_installed


def generate_recommendations(results):
    """Generate recommendations based on diagnostic results"""
    print_header("RECOMMENDATIONS", Colors.YELLOW)
    
    subsets_ready = [name for name, (ready, _) in results['subsets'].items() if ready]
    subsets_not_ready = [name for name, (ready, _) in results['subsets'].items() if not ready]
    
    if len(subsets_ready) == 0:
        print(f"{Colors.RED}⚠ No subsets are ready. Cannot run experiments yet.{Colors.ENDC}\n")
        print("Next steps:")
        print("1. Ensure config files exist in configs/ directory")
        print("2. Obtain or train ASTRA model weights for at least one subset")
        print("3. Download U-Net weights (shared across all subsets)")
        print(f"4. Run: python subset_utils.py --status")
        
    elif len(subsets_ready) < 5:
        print(f"{Colors.GREEN}✓ {len(subsets_ready)} subset(s) ready:{Colors.ENDC} {', '.join(subsets_ready)}\n")
        print(f"{Colors.YELLOW}⚡ {len(subsets_not_ready)} subset(s) not ready:{Colors.ENDC} {', '.join(subsets_not_ready)}\n")
        print("Options:")
        print(f"1. Run experiments on available subsets: {', '.join(subsets_ready)}")
        print(f"2. Work with author to obtain weights for: {', '.join(subsets_not_ready)}")
        print("3. Use transfer learning from ETH weights for missing subsets")
        
    else:
        print(f"{Colors.GREEN}✓ All subsets ready! You can run full experiments.{Colors.ENDC}\n")
        print("Suggested commands:")
        print("  # Check status")
        print("  python subset_utils.py --status")
        print("\n  # Run single experiment on all subsets")
        print("  python example_multi_subset_rank_ablation.py --subsets all")
        print("\n  # Run all experiments on all subsets")
        print("  python run_experiments_multi_subset.py --experiments all --subsets all")
    
    if not results['experiments']:
        print(f"\n{Colors.YELLOW}⚠ Some experiment scripts missing{Colors.ENDC}")
        print("Check experiment_scripts/ directory")
    
    if not results['utilities']:
        print(f"\n{Colors.YELLOW}⚠ Some utility files missing{Colors.ENDC}")
        print("Ensure all helper modules are present")
    
    if not results['dependencies']:
        print(f"\n{Colors.RED}⚠ Some Python packages not installed{Colors.ENDC}")
        print("Install missing packages with:")
        print("  pip install torch numpy pandas matplotlib pyyaml munch albumentations opencv-python")


def main():
    parser = argparse.ArgumentParser(
        description='Check ConformalASTRA setup for multi-subset experiments'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    parser.add_argument('--subset', '-s', type=str,
                       help='Check specific subset only')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("═"*70)
    print("  ConformalASTRA Multi-Subset Setup Diagnostic")
    print("═"*70)
    print(Colors.ENDC)
    
    results = {
        'subsets': {},
        'experiments': True,
        'utilities': True,
        'dependencies': True
    }
    
    # Check Python dependencies
    results['dependencies'] = check_dependencies()
    
    # Check utility files
    results['utilities'] = check_utilities(args.verbose)
    
    # Check experiment scripts
    results['experiments'] = check_experiment_scripts(args.verbose)
    
    # Check subsets
    print_header("SUBSET AVAILABILITY", Colors.BLUE)
    
    subsets = ['eth', 'hotel', 'univ', 'zara01', 'zara02']
    
    if args.subset:
        if args.subset not in subsets:
            print(f"{Colors.RED}Error: Unknown subset '{args.subset}'{Colors.ENDC}")
            print(f"Available: {', '.join(subsets)}")
            return
        subsets = [args.subset]
    
    for subset in subsets:
        ready, missing = check_subset(subset, args.verbose)
        results['subsets'][subset] = (ready, missing)
    
    # Generate recommendations
    generate_recommendations(results)
    
    # Summary
    print_header("SUMMARY", Colors.BLUE)
    
    ready_count = sum(1 for ready, _ in results['subsets'].values() if ready)
    total_count = len(results['subsets'])
    
    print(f"Subsets ready: {ready_count}/{total_count}")
    print(f"Experiments: {'✓' if results['experiments'] else '✗'}")
    print(f"Utilities: {'✓' if results['utilities'] else '✗'}")
    print(f"Dependencies: {'✓' if results['dependencies'] else '✗'}")
    
    overall_ready = (ready_count > 0 and 
                     results['experiments'] and 
                     results['utilities'] and 
                     results['dependencies'])
    
    if overall_ready:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ System ready for multi-subset experiments!{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ System not fully ready - see recommendations above{Colors.ENDC}")
    
    print()


if __name__ == "__main__":
    main()
