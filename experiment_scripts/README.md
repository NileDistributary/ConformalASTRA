# ConformalASTRA Experiment Scripts

## Baseline Settings
All experiments use the following baseline parameters (from castra3.py):
- split_ratio: 0.9 (90% calibration, 10% test)
- alpha: 0.1 (90% confidence)
- smallT: False
- past_window: 10
- stride: 1
- rank: None (when not varied)

## Experiment Structure

### Baseline Comparison
- `experiment_baseline_comparison.py`: Coordinate-wise vs MultiDimSPCI

### Ablation Studies
1. `experiment_ablation_rank.py`: Effect of covariance rank approximation
2. `experiment_ablation_calibration_size.py`: Effect of calibration set size
3. `experiment_ablation_qr_vs_empirical.py`: Quantile regression vs empirical
4. `experiment_ablation_past_window.py`: Effect of past window size
5. `experiment_ablation_global_vs_local.py`: Global vs local ellipsoids

## Running Experiments

### Individual Experiment
```bash
python experiment_scripts/experiment_ablation_rank.py
```

### All Experiments
```bash
python run_all_ablations.py
```

## Results Structure
- `results/csvs/`: CSV files with metrics
- `results/figures/`: Generated plots
- `results/configs/`: Experiment configurations
- `results/errors/`: Error logs (if any failures)

## Utilities
- `utils/experiment_utils.py`: Shared utility functions for all experiments
