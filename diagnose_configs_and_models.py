"""
Proper Config Comparison and Model Diagnostic

This script will:
1. Load and compare hotel vs ETH configs
2. Check what the models actually expect
3. Diagnose the "24 unexpected keys" issue
"""

import yaml
import torch


def load_and_display_config(config_path, name):
    """Load and display a config file"""
    print(f"\n{'='*70}")
    print(f"{name} CONFIG: {config_path}")
    print(f"{'='*70}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Display key settings
    print(f"\nDataset Settings:")
    print(f"  DATASET: {cfg.get('DATASET')}")
    print(f"  SUBSET: {cfg.get('SUBSET')}")
    
    print(f"\nModel Architecture:")
    model = cfg.get('MODEL', {})
    print(f"  USE_PRETRAINED_UNET: {model.get('USE_PRETRAINED_UNET')}")
    print(f"  USE_SOCIAL: {model.get('USE_SOCIAL')}")
    print(f"  USE_VAE: {model.get('USE_VAE')}")
    print(f"  FEATURE_EXTRACTOR: {model.get('FEATURE_EXTRACTOR')}")
    print(f"  FEATURE_DIM: {model.get('FEATURE_DIM')}")
    print(f"  ENC_LAYERS: {model.get('ENC_LAYERS')}")
    print(f"  DEC_LAYERS: {model.get('DEC_LAYERS')}")
    print(f"  D_MODEL: {model.get('D_MODEL')}")
    print(f"  NHEAD: {model.get('NHEAD')}")
    print(f"  DIM_FEEDFORWARD: {model.get('DIM_FEEDFORWARD')}")
    
    print(f"\nTraining Settings:")
    train = cfg.get('TRAIN', {})
    print(f"  DEVICE: {train.get('DEVICE')}")
    
    return cfg


def compare_configs(config1_path, config2_path, name1, name2):
    """Compare two config files"""
    print(f"\n{'='*70}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*70}")
    
    with open(config1_path, 'r') as f:
        cfg1 = yaml.safe_load(f)
    with open(config2_path, 'r') as f:
        cfg2 = yaml.safe_load(f)
    
    # Find differences
    differences = []
    
    def compare_dict(d1, d2, path=""):
        for key in set(list(d1.keys()) + list(d2.keys())):
            current_path = f"{path}.{key}" if path else key
            
            if key not in d1:
                differences.append(f"  {current_path}: Only in {name2} = {d2[key]}")
            elif key not in d2:
                differences.append(f"  {current_path}: Only in {name1} = {d1[key]}")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                compare_dict(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                differences.append(f"  {current_path}: {name1}={d1[key]} vs {name2}={d2[key]}")
    
    compare_dict(cfg1, cfg2)
    
    if differences:
        print(f"\nFound {len(differences)} differences:")
        for diff in differences:
            print(diff)
    else:
        print("\n⚠️  NO DIFFERENCES FOUND!")
        print("    This is WRONG - configs should at least differ in SUBSET!")
    
    return differences


def inspect_model_checkpoint(checkpoint_path, name):
    """Inspect what's actually in the checkpoint"""
    print(f"\n{'='*70}")
    print(f"{name} MODEL CHECKPOINT: {checkpoint_path}")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} parameters")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    state_dict = checkpoint.get('model_state_dict', {})
    
    print(f"\nParameter breakdown:")
    
    # Count parameters by module
    modules = {}
    for key in state_dict.keys():
        module = key.split('.')[0]
        if module not in modules:
            modules[module] = []
        modules[module].append(key)
    
    for module, keys in sorted(modules.items()):
        print(f"  {module}: {len(keys)} parameters")
    
    # Look for specific indicators
    has_unet = any('unet' in k.lower() for k in state_dict.keys())
    has_vae = any('vae' in k.lower() for k in state_dict.keys())
    has_social = any('social' in k.lower() or 'gcn' in k.lower() for k in state_dict.keys())
    
    print(f"\nFeature flags:")
    print(f"  Has U-Net params: {has_unet}")
    print(f"  Has VAE params: {has_vae}")
    print(f"  Has social params: {has_social}")
    
    return checkpoint, state_dict


def find_unexpected_keys(checkpoint_path, config_path):
    """Try to determine what the 24 unexpected keys are"""
    print(f"\n{'='*70}")
    print(f"IDENTIFYING UNEXPECTED KEYS")
    print(f"{'='*70}")
    
    # This requires loading ASTRA model, which is complex
    # But we can make educated guesses based on parameter names
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', {})
    
    print(f"\nTotal parameters in checkpoint: {len(state_dict)}")
    
    # Common sources of unexpected keys:
    print(f"\nPossible sources of unexpected keys:")
    
    # 1. VAE parameters (if USE_VAE was different during training)
    vae_params = [k for k in state_dict.keys() if 'vae' in k.lower() or 'latent' in k.lower() or 'kl' in k.lower()]
    print(f"  1. VAE-related: {len(vae_params)} parameters")
    if vae_params:
        print(f"     Examples: {vae_params[:3]}")
    
    # 2. Social/GCN parameters  
    social_params = [k for k in state_dict.keys() if 'social' in k.lower() or 'gcn' in k.lower() or 'graph' in k.lower()]
    print(f"  2. Social/GCN: {len(social_params)} parameters")
    if social_params:
        print(f"     Examples: {social_params[:3]}")
    
    # 3. U-Net parameters (if embedded in checkpoint)
    unet_params = [k for k in state_dict.keys() if 'unet' in k.lower() and 'embedding' not in k.lower()]
    print(f"  3. U-Net specific: {len(unet_params)} parameters")
    if unet_params:
        print(f"     Examples: {unet_params[:3]}")
    
    # 4. Embedding/feature extractor variations
    embed_params = [k for k in state_dict.keys() if 'embedding' in k.lower() or 'extractor' in k.lower()]
    print(f"  4. Embedding/Extractor: {len(embed_params)} parameters")
    
    # Print all parameter names (for debugging)
    print(f"\n{'='*70}")
    print(f"ALL PARAMETER NAMES (first 50):")
    print(f"{'='*70}")
    for i, key in enumerate(sorted(state_dict.keys())[:50]):
        print(f"{i+1:3d}. {key}")
    
    if len(state_dict) > 50:
        print(f"... and {len(state_dict) - 50} more")


def main():
    """Run all diagnostics"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL & CONFIG DIAGNOSTIC")
    print("="*70)
    
    # Step 1: Display both configs
    hotel_cfg = load_and_display_config('configs/hotel.yaml', 'HOTEL')
    eth_cfg = load_and_display_config('configs/eth.yaml', 'ETH')
    
    # Step 2: Compare configs
    diffs = compare_configs('configs/hotel.yaml', 'configs/eth.yaml', 'Hotel', 'ETH')
    
    # Step 3: Inspect checkpoints
    hotel_ckpt, hotel_state = inspect_model_checkpoint(
        'pretrained_astra_weights/hotel_best_model.pth', 'HOTEL')
    eth_ckpt, eth_state = inspect_model_checkpoint(
        'pretrained_astra_weights/eth_best_model.pth', 'ETH')
    
    # Step 4: Compare checkpoint sizes
    print(f"\n{'='*70}")
    print(f"CHECKPOINT COMPARISON")
    print(f"{'='*70}")
    print(f"\nParameter count:")
    print(f"  Hotel model: {len(hotel_state)} parameters")
    print(f"  ETH model: {len(eth_state)} parameters")
    print(f"  Difference: {len(hotel_state) - len(eth_state)}")
    
    # Step 5: Find unique keys
    hotel_keys = set(hotel_state.keys())
    eth_keys = set(eth_state.keys())
    
    only_hotel = hotel_keys - eth_keys
    only_eth = eth_keys - hotel_keys
    
    print(f"\nUnique parameters:")
    print(f"  Only in Hotel: {len(only_hotel)}")
    print(f"  Only in ETH: {len(only_eth)}")
    
    if only_hotel:
        print(f"\n  Parameters only in Hotel model:")
        for key in sorted(list(only_hotel))[:24]:  # Show first 24 (the unexpected keys!)
            print(f"    - {key}")
        if len(only_hotel) > 24:
            print(f"    ... and {len(only_hotel) - 24} more")
    
    if only_eth:
        print(f"\n  Parameters only in ETH model:")
        for key in sorted(list(only_eth))[:10]:
            print(f"    - {key}")
    
    # Step 6: Identify the 24 unexpected keys
    if len(only_hotel) == 24:
        print(f"\n{'='*70}")
        print(f"🎯 FOUND THE 24 UNEXPECTED KEYS!")
        print(f"{'='*70}")
        print(f"\nThese are the parameters in hotel_best_model.pth")
        print(f"that don't exist in the current ASTRA model structure:")
        for i, key in enumerate(sorted(only_hotel), 1):
            print(f"  {i:2d}. {key}")
    
    # Step 7: Analysis and recommendations
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    
    if not diffs:
        print("\n⚠️  CRITICAL: Configs are identical!")
        print("   They should differ at least in SUBSET parameter")
        print("   This might indicate configs were not properly set up")
    
    if len(only_hotel) > 0:
        print(f"\n⚠️  Hotel model has {len(only_hotel)} extra parameters")
        print("   Possible causes:")
        print("   1. Hotel was trained with different architecture (USE_VAE, USE_SOCIAL)")
        print("   2. Hotel is from different ASTRA version")
        print("   3. Hotel training used different config than configs/hotel.yaml")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print("\n1. Check if configs/hotel.yaml is the correct config:")
    print("   - Ask author which config was used for training")
    print("   - Compare with author's original hotel config")
    
    print("\n2. Check model architecture settings:")
    print("   - Verify USE_VAE setting matches training")
    print("   - Verify USE_SOCIAL setting matches training")
    print("   - These affect model structure")
    
    print("\n3. Test hypothesis:")
    print("   - Try USE_VAE=True in hotel config if currently False")
    print("   - Or vice versa")
    print("   - See if unexpected keys warning disappears")
    
    print("\n4. Contact author with this output:")
    print("   - Show them the 24 unexpected keys")
    print("   - Ask what config/architecture was used")
    
    print()


if __name__ == "__main__":
    main()
