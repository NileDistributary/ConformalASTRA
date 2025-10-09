import yaml
import munch
from pathlib import Path


class Config:
    """Simple configuration loader that mirrors patterns used in the repo.

    Usage:
        cfg = Config('configs/eth.yaml')
        print(cfg.MODEL.USE_PRETRAINED_UNET)
    """

    def __init__(self, cfg_path_or_dict):
        # Allow passing a path or an already-parsed dict/munch
        if isinstance(cfg_path_or_dict, (str, Path)):
            path = Path(cfg_path_or_dict)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            self._cfg = munch.munchify(data)
        elif isinstance(cfg_path_or_dict, dict):
            self._cfg = munch.munchify(cfg_path_or_dict)
        else:
            # assume already a munch or similar
            self._cfg = cfg_path_or_dict

        # Mirror attribute access to the internal munch object
    def __getattr__(self, name):
        return getattr(self._cfg, name)

    def __repr__(self):
        return f"Config({repr(self._cfg)})"
