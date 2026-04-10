# 🚀 DynamicRad: Adaptive Spatiotemporal Sparsity for Long Video Diffusion

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](#)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights_(Coming_Soon)-FFD21E)](#)

**[Yongji Long](#), [Shijun Liang](#), [Jintao Li](#), [Fengyu Sun](#), [Yun Li](#)**

*University of Electronic Science and Technology of China & Huawei Kirin Solution & Michigan State University*

</div>

---

**DynamicRad** is a unified sparse-attention paradigm that reconciles kernel-friendly structure with content adaptivity for long video diffusion models (e.g., Wan2.1-14B, HunyuanVideo). By introducing an **Offline Bayesian Optimization (BO) pipeline** and a **Semantic Motion Router**, DynamicRad pushes the efficiency-quality Pareto frontier, achieving **1.7×–2.5× inference speedups** with **over 80% effective sparsity** on NVIDIA H100 GPUs, all without the overhead of online neural architecture search.

![Teaser](assets/teaser_gif_placeholder.gif) 
*(Note: Visual comparisons between Dense Attention and DynamicRad on high-motion vs. low-motion scenes)*

## 🌟 News
* `[2026.04]` 🔥 Code and end-to-end inference scripts for **Wan2.1-14B** are officially released! 
* `[Coming Soon]` 📄 Our paper will be available on arXiv shortly.
* `[Coming Soon]` 🤗 Mask-Aware LoRA weights will be released on HuggingFace.
