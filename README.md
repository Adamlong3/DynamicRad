# 🚀 DynamicRad: Content-Adaptive Sparse Attention for Long Video Diffusion

<div align="center">

[![Conference Submission](https://img.shields.io/badge/Status-Under_Double--Blind_Review-blue.svg)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Anonymous Authors**

*Paper submitted for anonymous peer review.*

</div>

---

**DynamicRad** is a unified sparse-attention paradigm that reconciles kernel-friendly structure with content adaptivity for long video diffusion models (e.g., Wan2.1-14B and HunyuanVideo). By introducing an **Offline Bayesian Optimization (BO) pipeline** and a lightweight **Semantic Motion Router**, DynamicRad achieves a strong efficiency-quality trade-off, obtaining **1.7×–2.5× inference speedups** with **over 80% effective sparsity** on NVIDIA H100 GPUs, without the overhead of online neural architecture search.

---

## 🧠 Method Overview

<div align="center">
  <img src="assets/framework_overview.png" alt="DynamicRad Framework Overview" width="900"/>
</div>

*DynamicRad combines offline BO-based configuration, prompt-conditioned motion routing, a shared structured candidate set with dual-mode sparse selection, and an optional mask-aware LoRA refinement module.*

---

## 🎬 Qualitative Adaptivity

<div align="center">
  <img src="assets/qualitative_adaptivity.png" alt="DynamicRad Qualitative Adaptivity" width="900"/>
</div>

*DynamicRad automatically adapts its sparsity regime to the semantic motion implied by the prompt. For low-motion scenes, static-ratio mode produces highly sparse near-diagonal masks; for high-motion scenes, dynamic-threshold mode preserves long-range dependencies.*

---

## 🌟 News

- 🔥 Code and end-to-end inference scripts for **Wan2.1-14B** are released for anonymous peer review.
- 🔥 Offline BO profiling pipeline and plotting scripts are included for reproducibility.

---

## 🛠️ Installation & Environment Setup

DynamicRad is built on top of PyTorch/Diffusers and optimized sparse-attention kernels. We tested the code with Python 3.12, PyTorch 2.5.1 + CUDA 12.4, FlashInfer 0.5.1, and SageAttention 2.2.0.

### 1. Base Environment

```bash
conda create -n dynamicrad python=3.12 -y
conda activate dynamicrad

# PyTorch stack tested with CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install Dependencies

```bash
# Clone the anonymous repository
git clone <anonymous_repo_link>
cd DynamicRad

# Install general Python dependencies
pip install -r requirements.txt
```

### 3. Install Core Attention Kernels

To reproduce the reported speedups, DynamicRad relies on FlashInfer and SageAttention. Other CUDA/PyTorch versions may require matching kernel wheels.

```bash
# FlashInfer, tested with CUDA 12.4 and PyTorch 2.5
pip install flashinfer-python==0.5.1 -i https://flashinfer.ai/whl/cu124/torch2.5

# SageAttention
pip install sageattention==2.2.0
```

---

## 🚀 Quick Start (Inference)

We provide an end-to-end script to generate videos, visualize block-sparse masks, and run evaluation.

```bash
# Run the all-in-one pipeline
bash scripts/run_radial_vbench.sh
```

### Or run Python inference directly

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

A core contribution of DynamicRad is the **Offline BO pipeline**, which models spatiotemporal energy decay using a physics-grounded proxy task based on AR feature drift. The profiling pipeline can be re-run for new resolutions or hardware in **under 15 minutes**.

```bash
# Run 30 trials of TPE optimization across Low, Mid, and High motion regimes
python bo_pipeline/run_bo_pipeline.py --steps 30
```

This will automatically generate the lookup table used by the Semantic Motion Router, for example:

```text
final_bo_lookup_table_steps30.csv
```

### Visualizing BO Convergence

```bash
python scripts/plot_bo_convergence.py --steps 30
```

<div align="center">
  <img src="assets/bo_convergence.png" alt="BO Convergence" width="700"/>
</div>

*BO converges rapidly on the proxy task and produces motion-regime-specific configurations for Low, Mid, and High motion scenarios.*

---

## 📊 Main Results

DynamicRad achieves strong trade-offs between computational efficiency and generation quality, evaluated using **VisionReward** and **VBench** on **HunyuanVideo** and **Wan2.1-14B**.

<div align="center">
  <img src="assets/main_results.png" alt="DynamicRad Main Results" width="950"/>
</div>

*DynamicRad achieves 1.7×–2.5× speedups with over 80% effective sparsity. Static-ratio mode provides the highest throughput, while dynamic-threshold mode preserves or even improves quality in some long-sequence settings.*

---

## 📂 Code Structure

```text
DynamicRad/
├── attention/              # Core dual-mode sparse attention and mask generation
├── bo_pipeline/            # Offline BO proxy task, feature simulator, and LUT generation
├── models/
│   └── wan2_1/             # Monkey-patching scripts for Wan2.1-14B
├── scripts/                # End-to-end inference, evaluation, and plotting scripts
├── assets/                 # README figures and visualization assets
├── requirements.txt        # General Python dependencies
└── README.md
```

---

## 🖼️ Suggested Asset Filenames

Place the following files under `assets/`:

```text
assets/framework_overview.png
assets/qualitative_adaptivity.png
assets/bo_convergence.png
assets/main_results.png
```

Recommended correspondence:

- `framework_overview.png` → paper figure `fig:framework`
- `qualitative_adaptivity.png` → paper figure `fig:qualitative_vis`
- `bo_convergence.png` → BO convergence figure
- `main_results.png` → paper table `tab:main_results` exported as an image

---

## 🙏 Acknowledgements

This project builds upon several open-source efforts. We thank the developers of Radial Attention, FlashInfer, and Wan2.1 for releasing code and infrastructure that made this anonymous evaluation possible.

---

## 📑 Citation

*Citation details and deanonymized authors will be updated after the conclusion of the double-blind review process.*
