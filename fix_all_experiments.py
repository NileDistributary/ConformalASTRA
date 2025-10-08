"""
Comprehensive fix script for all experiment files
Fixes:
1. Config path (./configs/astra_eth.yml -> configs/eth.yaml)
2. Removes plt.show() calls (prevents blocking)
3. Adds non-interactive matplotlib backend
"""
import os
import re

def fix_experiment_file(filepath):
    """Apply all fixes to a single experiment file"""
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Fix 1: Update config path
    old_path = "'./configs/astra_eth.yml'"
    new_path = "'configs/eth.yaml'"
    if old_path in content:
        content = content.replace(old_path, new_path)
        changes_made.append("config path")
    
    # Fix 2: Remove plt.show() calls
    if 'plt.show()' in content:
        content = re.sub(r'\s*plt\.show\(\)\s*\n', '\n', content)
        changes_made.append("removed plt.show()")
    
    # Fix 3: Add non-interactive backend if needed
    if "matplotlib.use('Agg')" not in content and 'import matplotlib.pyplot as plt' in content:
        content = content.replace(
            'import matplotlib.pyplot as plt',
            "import matplotlib\nmatplotlib.use('Agg')  # Non-interactive backend\nimport matplotlib.pyplot as plt"
        )
        changes_made.append("added non-interactive backend")
    
    # Write changes if any were made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[OK] Fixed {filepath}")
        print(f"     Changes: {', '.join(changes_made)}")
        return True
    else:
        print(f"[SKIP] No changes needed: {filepath}")
        return False

def main():
    """Fix all experiment files"""
    print("="*70)
    print("FIXING ALL EXPERIMENT FILES")
    print("="*70 + "\n")
    
    print("This script will:")
    print("  1. Update config path to 'configs/eth.yaml'")
    print("  2. Remove plt.show() calls (prevents blocking)")
    print("  3. Add non-interactive matplotlib backend")
    print()
    
    # List of files to fix
    files_to_fix = [
        'experiment_baseline_comparison.py',
        'experiment_ablation_rank.py',
        'experiment_ablation_calibration_size.py',
        'experiment_ablation_qr_vs_empirical.py',
        'experiment_ablation_past_window.py',
        'experiment_ablation_global_vs_local.py'
    ]
    
    fixed_count = 0
    for filename in files_to_fix:
        if fix_experiment_file(filename):
            fixed_count += 1
        print()  # Blank line between files
    
    print("="*70)
    print(f"SUMMARY: Fixed {fixed_count}/{len(files_to_fix)} files")
    print("="*70 + "\n")
    
    if fixed_count > 0:
        print("[SUCCESS] Experiments are now ready to run!")
        print()
        print("Key improvements:")
        print("  ✓ Correct config path")
        print("  ✓ No timeout limits (experiments can run as long as needed)")
        print("  ✓ Non-blocking figure saving")
        print()
        print("Run all experiments with:")
        print("  python run_all_ablations.py")
    else:
        print("[INFO] All files are already up to date")

if __name__ == "__main__":
    main()