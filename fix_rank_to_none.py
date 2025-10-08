"""
Fix Script: Change default rank from 12 to None in all experiment scripts
"""
import os
import re

def fix_rank_in_file(filepath):
    """Change rank=12 to rank=None in experiment files"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    changes = []
    
    # Pattern 1: rank=12 in function parameters/calls
    content = re.sub(r'\brank=12\b', 'rank=None', content)
    if 'rank=12' in original:
        changes.append('rank=12 -> rank=None in parameters')
    
    # Pattern 2: Variable assignments like rank = 12
    content = re.sub(r'\brank\s*=\s*12\b', 'rank = None', content)
    
    # Pattern 3: In comments or documentation
    content = re.sub(r'rank:\s*12', 'rank: None', content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    return False, []

def main():
    print("="*70)
    print("FIXING DEFAULT RANK: 12 -> None")
    print("="*70 + "\n")
    
    # Files to fix
    experiment_files = [
        'experiment_scripts/experiment_baseline_comparison.py',
        'experiment_scripts/experiment_ablation_rank.py',
        'experiment_scripts/experiment_ablation_calibration_size.py',
        'experiment_scripts/experiment_ablation_qr_vs_empirical.py',
        'experiment_scripts/experiment_ablation_past_window.py',
        'experiment_scripts/experiment_ablation_global_vs_local.py'
    ]
    
    fixed_count = 0
    for filepath in experiment_files:
        if os.path.exists(filepath):
            changed, changes = fix_rank_in_file(filepath)
            if changed:
                print(f"[OK] Fixed: {filepath}")
                for change in changes:
                    print(f"     - {change}")
                fixed_count += 1
            else:
                print(f"[SKIP] No rank=12 found: {filepath}")
        else:
            print(f"[WARN] File not found: {filepath}")
    
    print(f"\n{'='*70}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*70}\n")
    
    print("Default rank is now None (full covariance matrix)")
    print("Rank ablation will test: [4, 8, 12, 16, 20, None]")

if __name__ == "__main__":
    main()