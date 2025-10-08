# 🚶‍♂️ ConformalASTRA: Uncertainty-Aware Trajectory Prediction

This repository extends [ASTRA](https://github.com/IzzeddinTeeti/ASTRA) — a **Scene-aware TRAnsformer-based model for trajectory prediction** — with post-hoc uncertainty quantification using **Conformal Prediction** techniques.

ASTRA is a lightweight pedestrian trajectory prediction model that integrates scene context, social interactions, and spatiotemporal dynamics. It generalizes across multiple perspectives and outperforms several baselines on the ETH-UCY and PIE datasets, while using ~7× fewer parameters than the leading competitor.

While ASTRA originally uses a Conditional Variational Auto-Encoder (CVAE) to model uncertainty, this pipeline replaces it with **distribution-free, statistically valid conformal prediction intervals**.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Download & Process Datasets](#download--process-datasets)
- [Download Pretrained Models](#download-pretrained-models)
- [Running the Pipeline](#running-the-pipeline)
- [Licensing](#licensing)
- [Citation](#citation)

---

## 🛠️ Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/NileDistributary/ConformalASTRA.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ConformalASTRA
   ```

---

## 🧪 Environment Setup
For this project, we made use of the Conda package and environment management system and it is advisable that you begin by first installing **Minconda** or Anaconda to use it as well.

⚠️ **Important:** If you're on a public or restricted Windows machine, **do not use PowerShell**.  
Instead, open the **Anaconda Prompt** (installed with Miniconda or Anaconda) and run all commands from there. This avoids permission issues and ensures Conda works correctly.

```bash
# Step 1: Create and activate Conda environment
conda create -n castra python=3.10
conda activate castra

# Step 2: Install GPU-enabled PyTorch stack
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Step 3: Install general dependencies
pip install -r general_requirements.txt

# Step 4: Manually install missing packages
pip install thop
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
pip install segmentation-models-pytorch
```

✅ These steps have been tested and verified to work with the ConformalASTRA pipeline on both Windows and Linux/Mac systems.

---

## 📦 Download & Process Datasets

> **ETH-UCY Dataset (Bird's Eye View - BEV)**  

**Linux/Mac:**
```bash
bash ./scripts/down_process_eth.bash
```

**Windows:**
```cmd
scripts\down_process_eth.bat
```

The dataset will be downloaded and structured as expected by ASTRA.

---

## 📥 Download Pretrained Models

> **U-Net Keypoint Embedding Model**

**Linux/Mac:**
```bash
bash ./scripts/down_pretrained_unet_models.bash
```

**Windows:**
```cmd
scripts\down_pretrained_unet_models.bat
```

Downloads pretrained U-Net weights into `./pretrained_unet_weights/`. These are used by default during ASTRA training.

> **ASTRA Model Weights**

**Linux/Mac:**
```bash
bash ./scripts/down_pretrained_astra_models.bash
```

**Windows:**
```cmd
scripts\down_pretrained_astra_models.bat
```

Downloads pretrained ASTRA weights into `./pretrained_astra_weights/`.

---

## 🚀 Running the Pipeline

Once setup is complete, run:

```bash
python astraspcivisual.py
```

This script integrates ASTRA's scene-aware predictions with conformal uncertainty intervals.

---

## 📝 Licensing

This repository integrates code from:

- **ASTRA** (© 2023 IzzeddinTeeti) — MIT License  
- **SPCI** (© 2024 hamrel-cxu) — MIT License  
- **ConformalASTRA Extensions** (© 2025 Nile Anderson) — MIT License  

See [LICENSE](./LICENSE) for full terms.

---

## 📖 Citation

If you use this work, please cite the original ASTRA and SPCI papers, and acknowledge this extension:

```bibtex
@inproceedings{astra2023,
  title={ASTRA: A Scene-aware Transformer-based Model for Trajectory Prediction},
  author={Izzeddin Teeti et al.},
  year={2023}
}

@inproceedings{xu2023SPCI,
  title = 	 {Sequential Predictive Conformal Inference for Time Series},
  author =       {Xu, Chen and Xie, Yao},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  year = 	 {2023}
}
```

---

## 🤝 Contributions

Pull requests welcome. For major changes, open an issue first to discuss your ideas.
