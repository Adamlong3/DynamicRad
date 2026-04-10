# 🚀 DynamicRad: Adaptive Spatiotemporal Sparsity for Long Video Diffusion

<div align="center">

[![Conference Submission](https://img.shields.io/badge/Status-Under_Double--Blind_Review-blue.svg)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Anonymous Authors**

*Paper submitted for anonymous peer review.*

</div>

---

**DynamicRad** is a unified sparse-attention paradigm that reconciles kernel-friendly structure with content adaptivity for long video diffusion models (e.g., Wan2.1-14B, HunyuanVideo). By introducing an **Offline Bayesian Optimization (BO) pipeline** and a **Semantic Motion Router**, DynamicRad pushes the efficiency-quality Pareto frontier, achieving **1.7×–2.5× inference speedups** with **over 80% effective sparsity** on NVIDIA H100 GPUs, all without the overhead of online neural architecture search.

![Teaser](assets/teaser_gif_placeholder.gif) 
*(Note: Visual comparisons between Dense Attention and DynamicRad on high-motion vs. low-motion scenes)*

## 🌟 News
* 🔥 Code and end-to-end inference scripts for **Wan2.1-14B** are officially released for peer review! 

---

## 🛠️ Installation & Environment Setup

DynamicRad is built upon standard FlashAttention-2 and highly optimized sparse kernels. 

### 1. Base Environment
```bash
conda create -n dynamicrad python=3.10 -y
conda activate dynamicrad
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==0.15.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. Install Dependencies
```bash
# Clone the anonymous repository
git clone <anonymous_repo_link>
cd DynamicRad

# Install basic requirements
pip install -r requirements.txt
```

### 3. Install Core Attention Kernels
To achieve extreme speedups (e.g., 0.37ms QK-scoring overhead), we rely on `flashinfer` and `sageattention`. Please install them carefully based on your CUDA version:
```bash
# Install FlashInfer (Example for CUDA 12.1, Torch 2.4)
pip install flashinfer -i [https://flashinfer.ai/whl/cu121/torch2.4](https://flashinfer.ai/whl/cu121/torch2.4)

# Install SageAttention (Optional but recommended for specific hardware architectures)
pip install sageattention==1.0.6
```

---

## 🚀 Quick Start (Inference)

We provide an easy-to-use, end-to-end bash script to generate videos, visualize block-sparse masks, and evaluate using VBench.

```bash
# Run the all-in-one pipeline script
bash scripts/run_radial_vbench.sh
```

**Or run Python inference directly:**
```bash
python scripts/inference_wan.py \
    --model_id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
    --prompt "FPV drone shot flying through a futuristic sci-fi tunnel at high speed..." \
    --pattern "radial" \
    --topk_mode "dynamic_threshold" \
    --block_size 32 \
    --mask_threshold 0.8
```

---

## ⚙️ Offline Bayesian Optimization (BO) Pipeline

A core contribution of DynamicRad is the Offline BO pipeline, which models spatiotemporal energy decay via a physics-grounded proxy task (AR feature drift). You can re-run the profiling for any custom resolution/hardware in **under 15 minutes**:

```bash
# Run 30 trials of TPE optimization across Low, Mid, and High motion regimes
python dynamicrad/bo_pipeline/run_bo_pipeline.py --steps 30
```
This will automatically generate the optimal `final_bo_lookup_table_steps30.csv` used by the Semantic Router.

### Visualizing Convergence
To reproduce the BO convergence and Pareto frontier figures from our submission:
```bash
python scripts/plot_bo_convergence.py --steps 30
python scripts/plot_pareto.py
```

---

## 📊 Evaluation & Results

DynamicRad achieves state-of-the-art trade-offs between computational efficiency and video quality (measured by VisionReward and VBench). 

<div align="center">
  <img src="assets/pareto_frontier.png" alt="Pareto Frontier" width="600"/>
</div>

*(Please refer to the main paper for comprehensive benchmarks and hardware latency breakdowns).*

---

## 📂 Code Structure
```text
DynamicRad/
├── dynamicrad/
│   ├── attention/          # Core dual-mode sparse attention & mask generation
│   └── bo_pipeline/        # Offline BO proxy task & feature simulator
├── models/
│   └── wan2_1/             # Monkey-patching scripts for Wan2.1-14B
├── scripts/                # End-to-end inference and plotting scripts
├── configs/                # Pre-computed BO Lookup Tables (LUT)
└── README.md
```

---

## 🙏 Acknowledgements
This project utilizes several open-source libraries. We deeply appreciate the developers of:
* [Radial Attention](https://github.com/mit-han-lab/radial-attention) for baseline configurations.
* [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for sparse attention kernels.
* [Wan2.1](https://github.com/Wan-Video/Wan2.1) for the diffusion backbone.

---

## 📑 Citation

*Citation details and deanonymized authors will be updated upon the conclusion of the double-blind review process.*
