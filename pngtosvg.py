#!/usr/bin/env python3
"""
Targeted PNG to SVG Converter
Only converts figure output calls (plt.savefig, fig.savefig)
Leaves data loading paths unchanged
"""
import os
import re

def convert_figure_outputs(filepath):
    """Convert only plt.savefig and fig.savefig calls from PNG to SVG"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        changes = []
        
        # Pattern 1: plt.savefig(...'.png', dpi=..., bbox_inches='tight')
        pattern1 = r"(plt\.savefig\([^)]*?\.png['\"])\s*,\s*dpi=\d+\s*,\s*bbox_inches=['\"]tight['\"]\s*\)"
        if re.search(pattern1, content):
            content = re.sub(pattern1, lambda m: m.group(1).replace('.png', '.svg') + ", bbox_inches='tight')", content)
            changes.append("plt.savefig with dpi parameter")
        
        # Pattern 2: fig.savefig(...'.png', dpi=..., bbox_inches='tight')
        pattern2 = r"(fig\.savefig\([^)]*?\.png['\"])\s*,\s*dpi=\d+\s*,\s*bbox_inches=['\"]tight['\"]\s*\)"
        if re.search(pattern2, content):
            content = re.sub(pattern2, lambda m: m.group(1).replace('.png', '.svg') + ", bbox_inches='tight')", content)
            changes.append("fig.savefig with dpi parameter")
        
        # Pattern 3: fig_path or save_path variable assignments ending in .png
        pattern3 = r"((?:fig_path|save_path)\s*=\s*f?['\"][^'\"]*?)\.png(['\"])"
        if re.search(pattern3, content):
            content = re.sub(pattern3, r"\1.svg\2", content)
            changes.append("fig_path/save_path variable")
        
        # Pattern 4: Print statements mentioning .png files in results/figures
        pattern4 = r"(results/figures/[^'\"]*?)\.png"
        if re.search(pattern4, content):
            content = re.sub(pattern4, r"\1.svg", content)
            changes.append("print statement paths")
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        return False, []
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False, []

def main():
    print("="*70)
    print("Targeted Figure Output Converter (PNG → SVG)")
    print("="*70)
    print("\nConverting figure output calls only...")
    print("(Data loading paths will be left unchanged)\n")
    
    # Files that need conversion based on diagnostic
    files_to_convert = [
        'baseline_hotel_quick.py',
        'castra.py',
        'compare_vae_to_conformal.py',
        'test.py',
        'vis.py'
    ]
    
    modified = []
    not_found = []
    no_changes = []
    
    for filepath in files_to_convert:
        if not os.path.exists(filepath):
            not_found.append(filepath)
            continue
        
        changed, modifications = convert_figure_outputs(filepath)
        if changed:
            modified.append((filepath, modifications))
            print(f"✓ Modified: {filepath}")
            for mod in modifications:
                print(f"    - {mod}")
        else:
            no_changes.append(filepath)
    
    # Report
    print(f"\n{'='*70}")
    print("Conversion Results")
    print(f"{'='*70}")
    print(f"Files successfully modified: {len(modified)}")
    print(f"Files skipped (no changes needed): {len(no_changes)}")
    print(f"Files not found: {len(not_found)}")
    
    if modified:
        print(f"\n✅ Modified {len(modified)} file(s):")
        for filepath, mods in modified:
            print(f"  • {filepath}")
    
    if no_changes:
        print(f"\n⚠️  No changes made to {len(no_changes)} file(s):")
        for filepath in no_changes:
            print(f"  • {filepath} (already SVG or different pattern)")
    
    if not_found:
        print(f"\n❌ Could not find {len(not_found)} file(s):")
        for filepath in not_found:
            print(f"  • {filepath}")
    
    print(f"\n{'='*70}")
    print("Next Steps:")
    print(f"{'='*70}")
    print("1. Run your experiments (e.g., python compare_vae_to_conformal.py)")
    print("2. Check results/figures/ for .svg files")
    print("3. Verify SVG files render correctly in browser")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()