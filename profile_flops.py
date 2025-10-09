import argparse
import json
import yaml
import munch
import torch
import sys

# Preload the real standard-library `profile` module into sys.modules to
# prevent name-shadowing when third-party libs import cProfile -> profile.
try:
    import sysconfig
    import importlib.machinery
    import importlib.util
    from pathlib import Path as _Path

    _stdlib = _Path(sysconfig.get_paths()['stdlib'])
    _stdlib_profile = _stdlib / 'profile.py'
    if _stdlib_profile.exists():
        loader = importlib.machinery.SourceFileLoader('profile', str(_stdlib_profile))
        spec = importlib.util.spec_from_loader('profile', loader)
        _profile_mod = importlib.util.module_from_spec(spec)
        loader.exec_module(_profile_mod)
        sys.modules['profile'] = _profile_mod
except Exception:
    pass

def compute_flops(cfg_path, batch_size=1, num_agents=1):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = munch.munchify(cfg_dict)

    # Defaults similar to other scripts
    cfg.device = 'cpu'
    cfg.device_list = []
    cfg.SUBSET = getattr(cfg, 'SUBSET', 'eth')

    try:
        obs_len = int(cfg.DATA.OBSERVE_LENGTH)
    except Exception:
        obs_time = cfg.PREDICTION.OBS_TIME if hasattr(cfg, 'PREDICTION') and hasattr(cfg.PREDICTION, 'OBS_TIME') else 3.2
        freq = cfg.DATA.FREQUENCY if hasattr(cfg, 'DATA') and hasattr(cfg.DATA, 'FREQUENCY') else 2.5
        obs_len = int(obs_time * freq)
    try:
        feat_dim = int(cfg.MODEL.FEATURE_DIM)
    except Exception:
        feat_dim = 512

    # Lazy import model to keep this script minimal
    from models.astra_model import ASTRA_model
    model = ASTRA_model(cfg).to('cpu')
    model.eval()

    # Create dummy inputs on CPU
    dummy_past = torch.randn(batch_size, num_agents, obs_len, 2)
    dummy_unet = torch.randn(batch_size, num_agents, obs_len, feat_dim)

    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_past, None, dummy_unet), verbose=False)
        flops = 2 * macs
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        return 1

    print(json.dumps({'macs': float(macs), 'flops': float(flops)}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/eth.yaml')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--num_agents', type=int, default=1)
    args = parser.parse_args()

    return compute_flops(args.config, batch_size=args.batch, num_agents=args.num_agents)


if __name__ == '__main__':
    sys.exit(main())
