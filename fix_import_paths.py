"""
Fix Script: Add missing 'import sys' and 'import os' before sys.path.insert()
"""
import os
import re

def fix_missing_imports(filepath):
    """Add import sys and import os before sys.path.insert()"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the issue exists
    if 'sys.path.insert' not in content:
        print(f"[SKIP] No sys.path.insert found: {os.path.basename(filepath)}")
        return False
    
    lines = content.split('\n')
    fixed_lines = []
    needs_fix = False
    
    for i, line in enumerate(lines):
        # Found sys.path.insert line
        if 'sys.path.insert' in line:
            # Check if import sys and import os are already above this line
            has_sys_import = any('import sys' in fixed_lines[j] for j in range(len(fixed_lines)))
            has_os_import = any('import os' in fixed_lines[j] for j in range(len(fixed_lines)))
            
            if not has_sys_import or not has_os_import:
                # Add the imports right before sys.path.insert
                if not has_sys_import:
                    fixed_lines.append('import sys')
                if not has_os_import:
                    fixed_lines.append('import os')
                fixed_lines.append('# Add parent directory to path so we can import from root')
                needs_fix = True
        
        fixed_lines.append(line)
    
    if needs_fix:
        new_content = '\n'.join(fixed_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    
    return False

def main():
    print("="*70)
    print("FIXING MISSING sys AND os IMPORTS")
    print("="*70 + "\n")
    
    print("Adding 'import sys' and 'import os' before sys.path.insert()\n")
    
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
            if fix_missing_imports(filepath):
                print(f"[OK] Fixed: {os.path.basename(filepath)}")
                fixed_count += 1
        else:
            print(f"[WARN] File not found: {filepath}")
    
    print(f"\n{'='*70}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*70}\n")
    
    if fixed_count > 0:
        print("✓ Added missing imports")
        print("✓ Files now have proper structure:")
        print()
        print('  """')
        print('  Docstring...')
        print('  """')
        print('  import sys')
        print('  import os')
        print('  # Add parent directory to path so we can import from root')
        print('  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))')
        print()
        print('  import numpy as np')
        print('  from data.eth import ETH_dataset')
        print('  ...')
        print()
        print("Ready to run: python run_all_ablations.py")

if __name__ == "__main__":
    main()