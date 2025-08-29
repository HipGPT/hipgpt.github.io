---
layout: base
title: Home
permalink: /
---

<h1 class="sr-only">HipGPT</h1>

<section class="hero">
  <img class="logo" src="/assets/images/hip-hamster.png" alt="HipGPT Logo">
  <p class="tagline">A lightweight GPT-2 style implementation in C++ & HIP</p>
<p class="hero-cta">
  <a class="btn" href="/getting-started/">🚀 Get Started</a>
  <a class="btn secondary" href="https://github.com/aarnetalman/hipgpt" target="_blank">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="18" height="18" style="vertical-align: middle; margin-right: 6px;">
      <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 
      6.53 5.47 7.59.4.07.55-.17.55-.38 
      0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
      -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 
      2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89
      -3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02
      .08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 
      2-.27.68 0 1.36.09 2 .27 1.53-1.04 
      2.2-.82 2.2-.82.44 1.1.16 1.92.08 
      2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 
      3.75-3.65 3.95.29.25.54.73.54 1.48 
      0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 
      8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg>
    View on GitHub
  </a>
</p>



</section>

# Welcome to HipGPT Documentation

This project provides a lightweight, from-scratch implementation of a GPT-2 style transformer model written in C++ and accelerated using AMD's **[HIP API](https://rocm.docs.amd.com/projects/HIP/en/latest/)** for ROCm-enabled GPUs.  

The project is designed to be a clear and understandable guide to the inner workings of large language models.

The code is intended for educational purposes and is not recommended for production use of any kind. 

--- 

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

---

### Documentation

- 🚀 [Getting Started](/getting-started/)  
- 🎓 [Training](/training/)  
- ✨ [Inference](/inference/)  
- 🏗 [Codebase](/codebase/)  

