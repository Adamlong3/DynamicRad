# 🚀 DynamicRad: Adaptive Spatiotemporal Sparsity for Long Video Diffusion

<div align="center">

[![Conference Submission](https://img.shields.io/badge/Status-Under_Double--Blind_Review-blue.svg)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Anonymous Authors**

*Paper submitted for anonymous peer review.*

</div>

---

**DynamicRad** is a unified sparse-attention paradigm that reconciles kernel-friendly structure with content adaptivity for long video diffusion models (e.g., Wan2.1-14B and HunyuanVideo). By introducing an **Offline Bayesian Optimization (BO) pipeline** and a lightweight **Semantic Motion Router**, DynamicRad pushes the efficiency-quality Pareto frontier, achieving **1.7×–2.5× inference speedups** with **over 80% effective sparsity** on NVIDIA H100 GPUs, without the overhead of online neural architecture search.

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

DynamicRad is built on top of standard FlashAttention-2 and highly optimized sparse kernels.

### 1. Base Environment

```bash
conda create -n dynamicrad python=3.10 -y
conda activate dynamicrad
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==0.15.2 --index-url https://download.pytorch.org/whl/cu121
