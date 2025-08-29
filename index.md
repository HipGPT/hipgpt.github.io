---
layout: base
title: Home
permalink: /
---

<p align="center">
  <img src="/assets/images/hip-hamster.png" alt="HipGPT Logo" width="200"/>
  <br/>
  <em>A lightweight GPT-2 style implementation in C++ & HIP</em>
</p>

<div class="model-card">
  <h3>📐 Default Model Parameters</h3>
  <p><strong>Recommended small config:</strong> ~28M params</p>
  <ul>
    <li><b>E</b> = 256 (embedding size)</li>
    <li><b>L</b> = 8 (layers)</li>
    <li><b>H</b> = 8 (attention heads)</li>
    <li><b>F</b> = 1024 (feed-forward size)</li>
    <li><b>V</b> ≈ 5k (vocabulary size)</li>
    <li><b>S</b> = 256 (sequence length)</li>
  </ul>
</div>

<p align="center">
  <a href="https://github.com/aarnetalman/hipgpt" target="_blank">
    <img src="https://img.shields.io/badge/View_on_GitHub-hipgpt-black?logo=github&style=for-the-badge"/>
  </a>
</p>


# Welcome to HipGPT Documentation

This project provides a lightweight, from-scratch implementation of a GPT-2 style transformer model written in C++ and accelerated using AMD's **[HIP API](https://rocm.docs.amd.com/projects/HIP/en/latest/)** for ROCm-enabled GPUs. The project is designed to be a clear and understandable guide to the inner workings of large language models.

The code is intended for educational purposes and is not recommended for production use of any kind. 

### Key Features:

* **Custom BPE Tokenizer:** A Byte-Pair Encoding tokenizer built from scratch that can be trained on any raw text file. The `Tokenizer` class handles converting raw text into integer token IDs and back.
* **GPT Model Architecture:** A decoder-only transformer inspired by GPT-2. The `GPTModel` class manages token/positional embeddings, stacks multiple `TransformerLayer` instances, and includes a final linear layer to produce output logits over the vocabulary.  
  * Includes **LayerNorm** before attention and feed-forward sublayers  
  * Supports **dropout** during training  
* **HIP Acceleration:** All performance-critical operations (matrix multiplication, attention, layer normalization, etc.) are implemented with custom HIP kernels for AMD GPUs.  
  * Includes optimized **FlashAttention kernels** (for head dimensions 32/64) with safe fallback implementations  
* **Flexible Sampling:** Text generation supports top-k, top-p (nucleus), temperature, end-of-sequence tokens, and repetition penalty.  
* **End-to-End Workflow:** Includes scripts for downloading data, training from scratch, and generating new text.  
* **Self-Contained Build System:** Build managed by CMake, which fetches dependencies automatically.

### Documentation

- 🚀 [Getting Started](/getting-started/)  
- 🎓 [Training](/training/)  
- ✨ [Inference](/inference/)  
- 🏗 [Codebase](/codebase/)  

### Project Structure

For a deeper look into the codebase, see the [Codebase](/codebase/) section:

* `include/`: Public header files (model, kernels, tokenizer, transformer layers)  
* `src/`: C++ source implementations and CLI entrypoints  
* `scripts/`: Helper scripts for dataset prep, training, and inference  
