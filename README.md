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
