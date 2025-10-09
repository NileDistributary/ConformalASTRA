"""
ASTRA Model Profiling Implementation (moved out of `profile.py` to avoid name clash
with the standard-library `profile` module).
"""
# Ensure the real standard-library `profile` module is available under the
# name 'profile' before importing heavy libraries that may import cProfile.
try:
    import sysconfig
    import importlib.machinery
    import importlib.util
    from pathlib import Path as _Path
    import sys as _sys

    _stdlib = _Path(sysconfig.get_paths()['stdlib'])
    _stdlib_profile = _stdlib / 'profile.py'
    if _stdlib_profile.exists():
        loader = importlib.machinery.SourceFileLoader('profile', str(_stdlib_profile))
        spec = importlib.util.spec_from_loader('profile', loader)
        _profile_mod = importlib.util.module_from_spec(spec)
        loader.exec_module(_profile_mod)
        _sys.modules['profile'] = _profile_mod
except Exception:
    # If anything fails here, we'll continue — the launcher will still try to
    # run the implementation in a fresh process and lazy imports may avoid
    # circular issues. Report nothing here to avoid noisy output.
    pass

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from thop import profile, clever_format
from collections import OrderedDict
import json

import yaml
import munch
from utils.logger import get_logger
from icecream import ic
ic.disable()

logger = get_logger(__name__)


class ModelProfiler:
    """Comprehensive model profiling for ASTRA variants"""
    
    def __init__(self, device='cuda', warmup_runs=10, test_runs=100):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.results = {}
        
    def profile_baseline_astra(self, cfg, checkpoint_path=None):
        """Profile baseline ASTRA model"""
        print("\n" + "="*70)
        print("Profiling Baseline ASTRA")
        print("="*70)
        # Lazy import to avoid circular import with stdlib `profile`/cProfile
        from models.astra_model import ASTRA_model

        # Ensure the config exposes a device attribute expected by models
        # Some code (e.g. models/astra_model.py) expects `cfg.device` to exist.
        # If it's missing, set a sensible default based on this profiler's device.
        try:
            _ = cfg.device
        except Exception:
            # Munch raises AttributeError/KeyError when missing; set fallback
            try:
                cfg.device = self.device
            except Exception:
                # As a last resort, attach to the underlying dict if present
                try:
                    if isinstance(cfg, dict):
                        cfg['device'] = self.device
                except Exception:
                    pass
        # Ensure device_list is present (some modules use len(cfg.device_list))
        try:
            _ = cfg.device_list
        except Exception:
            try:
                cfg.device_list = [cfg.device] if cfg.device != 'cpu' else []
            except Exception:
                try:
                    if isinstance(cfg, dict):
                        cfg['device_list'] = [cfg.get('device', self.device)] if cfg.get('device', self.device) != 'cpu' else []
                except Exception:
                    pass

        # Ensure SUBSET exists for dataset-specific logic
        try:
            _ = cfg.SUBSET
        except Exception:
            try:
                cfg.SUBSET = 'eth'
            except Exception:
                try:
                    if isinstance(cfg, dict):
                        cfg['SUBSET'] = 'eth'
                except Exception:
                    pass

        # Ensure observation/prediction lengths expected by the model exist.
        # The repo config commonly stores continuous times under PREDICTION and
        # a sampling FREQUENCY under DATA. The model expects integer lengths
        # like cfg.DATA.OBSERVE_LENGTH and cfg.DATA.PREDICT_LENGTH.
        try:
            _ = cfg.DATA.OBSERVE_LENGTH
        except Exception:
            try:
                obs_time = cfg.PREDICTION.OBS_TIME if hasattr(cfg, 'PREDICTION') and hasattr(cfg.PREDICTION, 'OBS_TIME') else 3.2
                freq = cfg.DATA.FREQUENCY if hasattr(cfg, 'DATA') and hasattr(cfg.DATA, 'FREQUENCY') else 2.5
                cfg.DATA.OBSERVE_LENGTH = int(obs_time * freq)
            except Exception:
                try:
                    if isinstance(cfg, dict):
                        cfg.setdefault('DATA', {})
                        cfg['DATA']['OBSERVE_LENGTH'] = int(cfg.get('PREDICTION', {}).get('OBS_TIME', 3.2) * cfg['DATA'].get('FREQUENCY', 2.5))
                except Exception:
                    pass

        try:
            _ = cfg.DATA.PREDICT_LENGTH
        except Exception:
            try:
                # ASTRA_model uses math.ceil for predict length; mirror that here
                import math as _math
                pred_time = cfg.PREDICTION.PRED_TIME if hasattr(cfg, 'PREDICTION') and hasattr(cfg.PREDICTION, 'PRED_TIME') else 4.6
                freq = cfg.DATA.FREQUENCY if hasattr(cfg, 'DATA') and hasattr(cfg.DATA, 'FREQUENCY') else 2.5
                cfg.DATA.PREDICT_LENGTH = int(_math.ceil(pred_time * freq))
            except Exception:
                try:
                    if isinstance(cfg, dict):
                        cfg.setdefault('DATA', {})
                        cfg['DATA']['PREDICT_LENGTH'] = int(__import__('math').ceil(cfg.get('PREDICTION', {}).get('PRED_TIME', 4.6) * cfg['DATA'].get('FREQUENCY', 2.5)))
                except Exception:
                    pass

        # Initialize model
        model = ASTRA_model(cfg).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        
        model.eval()
        
        # Create dummy input
        batch_size = 1
        num_agents = 1
        obs_len = cfg.DATA.OBSERVE_LENGTH
        pred_len = cfg.DATA.PREDICT_LENGTH
        input_dim = 2  # x, y coordinates
        
        dummy_past = torch.randn(batch_size, num_agents, obs_len, input_dim).to(self.device)
        
        # Add U-Net features if used. Prefer to mirror `castra.py` behavior by
        # running a UNETEmbeddingExtractor on dummy images so shapes and
        # feature extraction logic match the real pipeline.
        if cfg.MODEL.USE_PRETRAINED_UNET:
            try:
                # Lazy-import the extractor used in the real scripts
                from models.keypoint_model import UNETEmbeddingExtractor

                embedding_extractor = UNETEmbeddingExtractor(cfg)
                embedding_extractor.eval()
                embedding_extractor = embedding_extractor.to(self.device)

                # Optionally load pretrained UNET weights if available (non-fatal)
                unet_weights_path = Path(f'pretrained_unet_weights/{cfg.SUBSET}_unet_model_best.pt')
                if unet_weights_path.exists():
                    try:
                        embedding_extractor.load_state_dict(torch.load(unet_weights_path, map_location=self.device))
                    except Exception:
                        # ignore loading errors for profiling
                        pass

                # Create dummy images shaped like the real pipeline expects.
                # castra.py flattens imgs to (-1, 3, H, W) before passing to the
                # extractor. We'll create random images and run them through the
                # extractor, then reshape the extracted features back to
                # (batch, agents, frames, feat_dim).
                try:
                    reshape_size = int(cfg.DATA.MIN_RESHAPE_SIZE)
                except Exception:
                    reshape_size = 224

                # match castra: imgs.view(-1, 3, H, W) where -1 == batch*frames*agents
                num_images = batch_size * obs_len * num_agents
                dummy_images = torch.randn(num_images, 3, reshape_size, reshape_size, device=self.device)

                with torch.no_grad():
                    _, _, extracted_features = embedding_extractor(dummy_images)

                # extracted_features: (N, feat_dim) -> reshape to (batch, agents, frames, feat_dim)
                feat_dim = extracted_features.shape[-1]
                dummy_unet = extracted_features.view(batch_size, num_agents, obs_len, feat_dim)
            except Exception:
                # Fall back to a random tensor if anything in the extractor path fails
                try:
                    unet_feature_dim = int(cfg.MODEL.FEATURE_DIM) if hasattr(cfg.MODEL, 'FEATURE_DIM') else 512
                except Exception:
                    unet_feature_dim = 512
                dummy_unet = torch.randn(batch_size, num_agents, obs_len, unet_feature_dim).to(self.device)
        else:
            dummy_unet = None
        
        # Profile parameters count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Profile FLOPs and MACs
        with torch.no_grad():
            if cfg.MODEL.USE_VAE:
                # For stochastic prediction (fut_loc is None in testing mode)
                input_tuple = (dummy_past, None, dummy_unet)
            else:
                # For deterministic prediction: pass a placeholder `fut_loc` (None)
                input_tuple = (dummy_past, None, dummy_unet)
            
            # THOP will register buffers named 'total_ops'/'total_params' on
            # modules; if we pre-create normal Python attributes with those
            # names, register_buffer will raise a KeyError. Ensure any such
            # attributes are removed so THOP can register its buffers.
            for m in model.modules():
                if hasattr(m, 'total_ops'):
                    try:
                        delattr(m, 'total_ops')
                    except Exception:
                        # try removing from buffers if somehow present
                        try:
                            if hasattr(m, '_buffers') and 'total_ops' in m._buffers:
                                del m._buffers['total_ops']
                        except Exception:
                            pass
                if hasattr(m, 'total_params'):
                    try:
                        delattr(m, 'total_params')
                    except Exception:
                        try:
                            if hasattr(m, '_buffers') and 'total_params' in m._buffers:
                                del m._buffers['total_params']
                        except Exception:
                            pass

            # Attempt to compute MACs/FLOPs in a separate Python process to
            # isolate THOP's hooks and avoid conflicts with in-process
            # monkeypatches. This mirrors how the working scripts run the
            # model without THOP, but allows FLOP counting when possible.
            try:
                import subprocess
                import shlex
                cmd = f"{sys.executable} \"{Path('profile_flops.py').absolute()}\" --config \"{Path('configs/eth.yaml').absolute()}\" --batch 1 --num_agents 1"
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                out = proc.stdout.strip() or proc.stderr.strip()
                try:
                    import json as _json
                    data = _json.loads(out)
                    macs = float(data.get('macs', 0))
                    flops = float(data.get('flops', 0))
                    print(f"✓ External FLOPs worker returned: MACs={macs}, FLOPs={flops}")
                except Exception:
                    print(f"⚠️ FLOPs subprocess did not return JSON: {out}")
                    macs = 0
                    flops = 0
            except Exception as e:
                print(f"⚠️ Failed to run external FLOPs worker: {e}")
                macs = 0
                flops = 0
        
        # Profile inference time
        inference_times = []
        
        # Warmup
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                if cfg.MODEL.USE_VAE:
                    _ = model(dummy_past, None, dummy_unet, mode='testing')
                else:
                    _ = model(dummy_past, None, dummy_unet)
        
        # Actual timing
        torch.cuda.synchronize() if self.device == 'cuda' else None
        for _ in range(self.test_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                if cfg.MODEL.USE_VAE:
                    _ = model(dummy_past, None, dummy_unet, mode='testing')
                else:
                    _ = model(dummy_past, None, dummy_unet)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        std_inference_time = np.std(inference_times) * 1000
        
        # Format results
        macs_formatted, flops_formatted = clever_format([macs, flops], "%.3f")
        
        baseline_results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'macs': macs,
            'macs_formatted': macs_formatted,
            'flops': flops,
            'flops_formatted': flops_formatted,
            'inference_time_ms': avg_inference_time,
            'inference_std_ms': std_inference_time,
            'uses_vae': cfg.MODEL.USE_VAE,
            'uses_unet': cfg.MODEL.USE_PRETRAINED_UNET
        }
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def profile_conformal_astra(self, cfg, X_train, Y_train, X_test, Y_test, alpha=0.1):
        """Profile Conformal Prediction Modified ASTRA"""
        print("\n" + "="*70)
        print("Profiling Conformal Prediction Modified ASTRA")
        print("="*70)
        
        from astra_wrapper import ASTRASklearnWrapper as ASTRAWrapper
        from helpers.MultiDim_SPCI_class import SPCI_and_EnbPI as MultiDimSPCI_and_EnbPI

        # Initialize ASTRA wrapper (uses astra_wrapper.ASTRASklearnWrapper)
        try:
            astra_wrapper = ASTRAWrapper(config_path=str(cfg.get('config_path', 'configs/eth.yaml')))
        except Exception:
            astra_wrapper = ASTRAWrapper()
        
        # Profile base ASTRA inference
        base_results = self.profile_baseline_astra(cfg)
        
        # Initialize MultiDimSPCI
        X_train_flat = X_train['past_trajectories'].reshape(len(X_train['past_trajectories']), -1)
        X_test_flat = X_test['past_trajectories'].reshape(len(X_test['past_trajectories']), -1)
        
        spci = MultiDimSPCI_and_EnbPI(
            X_train=X_train_flat,
            X_predict=X_test_flat,
            Y_train=Y_train,
            Y_predict=Y_test,
            fit_func=astra_wrapper
        )
        
        # Profile conformal prediction components using the same pipeline as castra.py
        print("\n📊 Profiling Conformal Prediction Components (castra settings):")

        # 1) Predictions on calibration/train and test sets
        start_time = time.perf_counter()
        Y_pred_train = astra_wrapper.predict(X_train)
        Y_pred_test = astra_wrapper.predict(X_test)
        residuals_calib = Y_train - Y_pred_train
        residuals_test = Y_test - Y_pred_test
        pred_time_ms = (time.perf_counter() - start_time) * 1000

        # Populate spci internal state exactly like castra.py
        n_calib = len(residuals_calib)
        n_test = len(residuals_test)
        spci.Ensemble_train_interval_centers = Y_pred_train
        spci.Ensemble_pred_interval_centers = Y_pred_test
        spci.Ensemble_online_resid[:n_calib] = residuals_calib
        spci.Ensemble_online_resid[n_calib:n_calib+n_test] = residuals_test

        # 2) Compute nonconformity scores (this also computes covariance in spci)
        start_time = time.perf_counter()
        spci.get_test_et = False
        spci.train_et = spci.get_et(residuals_calib)
        spci.get_test_et = True
        spci.test_et = spci.get_et(residuals_test)
        spci.all_et = np.concatenate([spci.train_et, spci.test_et])
        cov_time = 0.0
        svd_time = 0.0
        covariance_matrix = getattr(spci, 'global_cov', None)
        if covariance_matrix is None:
            covariance_matrix = np.cov(residuals_calib.T)
        try:
            start_svd = time.perf_counter()
            U, S, Vt = np.linalg.svd(covariance_matrix)
            svd_time = (time.perf_counter() - start_svd) * 1000
        except Exception:
            svd_time = 0.0
        cov_time = (time.perf_counter() - start_time) * 1000
        covariance_memory = covariance_matrix.nbytes / (1024 * 1024) if covariance_matrix is not None else 0.0

        # 3) Run the full SPCI quantile-regression pipeline as castra.py does
        start_time = time.perf_counter()
        try:
            spci.compute_Widths_Ensemble_online(alpha=alpha, stride=1, smallT=False, past_window=10, use_SPCI=True)
            qrf_train_time_ms = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            qrf_train_time_ms = (time.perf_counter() - start_time) * 1000
            print(f"⚠️ compute_Widths_Ensemble_online failed: {e}")

        # 4) Quantile estimator parameter accounting: inspect spci.QRF_ls or spci.rfqr
        import pickle
        quantile_exact_params = 0
        quantile_serialized_bytes = 0
        qrf_models_count = 0
        try:
            qrf_list = getattr(spci, 'QRF_ls', None)
            if not qrf_list and hasattr(spci, 'rfqr') and spci.rfqr is not None:
                qrf_list = [spci.rfqr]
            if qrf_list:
                for rf in qrf_list:
                    qrf_models_count += 1
                    nodes = 0
                    for est in getattr(rf, 'estimators_', []):
                        try:
                            nodes += int(est.tree_.node_count)
                        except Exception:
                            pass
                    quantile_exact_params += nodes * max(1, getattr(spci, 'd', residuals_calib.shape[1]))
                    try:
                        quantile_serialized_bytes += len(pickle.dumps(rf))
                    except Exception:
                        pass
        except Exception:
            quantile_exact_params = 0

        # Fallback to previous proxy if nothing was found
        if quantile_exact_params == 0:
            try:
                quantile_exact_params = int(qrf_params_estimate)
            except Exception:
                quantile_exact_params = 0

        # 5) Quick subset prediction + scoring to time the conformal overhead
        torch.cuda.synchronize() if self.device == 'cuda' else None
        small_n = min(10, len(X_test['past_trajectories']))
        X_test_subset = {'past_trajectories': X_test['past_trajectories'][:small_n]}
        start_time = time.perf_counter()
        Y_pred_subset = astra_wrapper.predict(X_test_subset)
        test_residuals = Y_test[:small_n] - Y_pred_subset
        spci.get_test_et = False
        scores = spci.get_et(test_residuals)
        torch.cuda.synchronize() if self.device == 'cuda' else None
        conformal_overhead = (time.perf_counter() - start_time) * 1000 + qrf_train_time_ms
        
        conformal_results = {
            'base_model': base_results,
            'covariance_computation_ms': cov_time,
            'svd_decomposition_ms': svd_time,
            'covariance_matrix_shape': covariance_matrix.shape,
            'covariance_memory_mb': covariance_memory,
            'rank_approximation': getattr(spci, 'r', 'full'),
            'quantile_estimator': {
                'method': 'Quantile Random Forest (SPCI)',
                'fitted_models': qrf_models_count,
                'serialized_bytes': quantile_serialized_bytes
            },
            'quantile_estimator_params': quantile_exact_params,
            'conformal_overhead_ms': conformal_overhead,
            'total_inference_ms': base_results['inference_time_ms'] + conformal_overhead,
            'alpha': alpha
        }
        
        self.results['conformal'] = conformal_results
        return conformal_results

    def generate_comparison_table(self):
        if 'baseline' not in self.results or 'conformal' not in self.results:
            print("⚠️ Both models must be profiled first")
            return None
        baseline = self.results['baseline']
        conformal = self.results['conformal']
        comparison = pd.DataFrame({
            'Metric': [
                'Total Parameters',
                'Trainable Parameters',
                'FLOPs',
                'MACs',
                'Base Inference Time (ms)',
                'Conformal Overhead (ms)',
                'Total Inference Time (ms)',
                'Covariance Memory (MB)',
                'Quantile Estimator Params',
            ],
            'Baseline ASTRA': [
                f"{baseline['total_parameters']:,}",
                f"{baseline['trainable_parameters']:,}",
                baseline['flops_formatted'],
                baseline['macs_formatted'],
                f"{baseline['inference_time_ms']:.3f} ± {baseline['inference_std_ms']:.3f}",
                "N/A",
                f"{baseline['inference_time_ms']:.3f}",
                "N/A",
                "N/A"
            ],
            'Conformal ASTRA': [
                f"{baseline['total_parameters']:,}",
                f"{baseline['trainable_parameters']:,}",
                baseline['flops_formatted'],
                baseline['macs_formatted'],
                f"{baseline['inference_time_ms']:.3f}",
                f"{conformal['conformal_overhead_ms']:.3f}",
                f"{conformal['total_inference_ms']:.3f}",
                f"{conformal['covariance_memory_mb']:.2f}",
                f"{conformal['quantile_estimator_params']:,}"
            ]
        })
        print("\n" + "="*70)
        print("MODEL COMPARISON TABLE")
        print("="*70)
        print(comparison.to_string(index=False))
        print("="*70)
        return comparison

    def save_results(self, output_dir='profiling_results'):
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = Path(output_dir) / f'profiling_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        if 'baseline' in self.results and 'conformal' in self.results:
            comparison = self.generate_comparison_table()
            csv_path = Path(output_dir) / f'comparison_table_{timestamp}.csv'
            comparison.to_csv(csv_path, index=False)
            txt_path = Path(output_dir) / f'profiling_summary_{timestamp}.txt'
            with open(txt_path, 'w') as f:
                f.write("ASTRA MODEL PROFILING SUMMARY\n")
                f.write("="*70 + "\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Test runs: {self.test_runs}\n\n")
                f.write(comparison.to_string(index=False))
                f.write("\n\n" + "="*70 + "\n")
                f.write("KEY INSIGHTS:\n")
                f.write(f"• Conformal overhead: {self.results['conformal']['conformal_overhead_ms']:.2f} ms\n")
                f.write(f"• Memory overhead: {self.results['conformal']['covariance_memory_mb']:.2f} MB\n")
                f.write(f"• Total slowdown: {self.results['conformal']['total_inference_ms']/self.results['baseline']['inference_time_ms']:.2f}x\n")
        print(f"\n✓ Results saved to {output_dir}/")
        return output_dir


def main():
    # Try to load config
    cfg_path = Path('configs/ASTRA/ASTRA.yml')
    if not cfg_path.exists():
        cfg_path = Path('configs/eth.yaml')
    if not cfg_path.exists():
        raise FileNotFoundError("Could not find a configuration file at 'configs/ASTRA/ASTRA.yml' or 'configs/eth.yaml'. Please provide a valid config path.")
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = munch.munchify(cfg_dict)

    # Provide defaults similar to `castra.py` so examples run without manual edits.
    # Many modules (models, wrappers) expect `cfg.device` and `cfg.device_list` to exist.
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        # if cfg already has these attributes, leave them alone
        _ = cfg.device
    except Exception:
        cfg.device = default_device
    try:
        _ = cfg.device_list
    except Exception:
        cfg.device_list = [cfg.device] if cfg.device != 'cpu' else []
    try:
        _ = cfg.SUBSET
    except Exception:
        cfg.SUBSET = 'eth'

    profiler = ModelProfiler(device=cfg.device, warmup_runs=10, test_runs=100)
    print("\n STARTING ASTRA MODEL PROFILING \n" + "="*70)
    checkpoint_path = 'checkpoints/ASTRA_ETH.pt'
    baseline_results = profiler.profile_baseline_astra(cfg, checkpoint_path)
    print("\n✓ Baseline ASTRA Profiling Complete:")
    print(f"  • Parameters: {baseline_results['total_parameters']:,}")
    print(f"  • FLOPs: {baseline_results['flops_formatted']}")
    print(f"  • MACs: {baseline_results['macs_formatted']}")
    print(f"  • Inference: {baseline_results['inference_time_ms']:.3f} ms")

    # Create dummy data with correct shapes expected by ASTRASklearnWrapper
    n_samples_train, n_samples_test = 100, 20
    obs_len = cfg.DATA.OBSERVE_LENGTH
    # Ensure predict length uses same ceiling logic as the model
    try:
        pred_len = int(cfg.DATA.PREDICT_LENGTH)
    except Exception:
        pred_len = int(np.ceil(cfg.PREDICTION.PRED_TIME * cfg.DATA.FREQUENCY))
    n_agents = 1

    # past_trajectories shape: (n_samples, n_agents, obs_len, 2)
    X_train = {'past_trajectories': np.random.randn(n_samples_train, n_agents, obs_len, 2)}
    Y_train = np.random.randn(n_samples_train, n_agents * pred_len * 2).reshape(n_samples_train, -1)
    X_test = {'past_trajectories': np.random.randn(n_samples_test, n_agents, obs_len, 2)}
    Y_test = np.random.randn(n_samples_test, n_agents * pred_len * 2).reshape(n_samples_test, -1)

    conformal_results = profiler.profile_conformal_astra(cfg, X_train, Y_train, X_test, Y_test, alpha=0.1)
    print("\n✓ Conformal ASTRA Profiling Complete:")
    print(f"  • Conformal overhead: {conformal_results['conformal_overhead_ms']:.3f} ms")
    print(f"  • Total inference: {conformal_results['total_inference_ms']:.3f} ms")
    print(f"  • Covariance memory: {conformal_results['covariance_memory_mb']:.2f} MB")

    profiler.generate_comparison_table()
    profiler.save_results()


if __name__ == '__main__':
    main()
