---
layout: base
title: Home
permalink: /
---

<p align="center">
  <img src="/assets/images/hip-hamster.png" alt="HipGPT Logo" width="200"/>
  <br/>
  <em>A lightweight GPT-2 implementation in C++ & HIP</em>
</p>

<p align="center">
  <b>Model size (default):</b> ~1.55M params  
  <sub>(E=128, L=2, H=4, F=256, V≈5k, S=32)</sub>
</p>

# Welcome to HipGPT Documentation

This project provides a lightweight, from-scratch implementation of a GPT-2 style transformer model written in C++ and accelerated using AMD's **[HIP API](https://rocm.docs.amd.com/projects/HIP/en/latest/)** for ROCm-enabled GPUs. The project is designed to be a clear and understandable guide to the inner workings of large language models.

The code is intended for educational purposes and is not recommended for production use of any kind. 

<p align="center">
  <a href="https://github.com/aarnetalman/hipgpt" target="_blank">
    <img src="https://img.shields.io/badge/View_on_GitHub-hipgpt-black?logo=github&style=for-the-badge"/>
  </a>
</p>


### Key Features:

  * **Custom BPE Tokenizer:** A Byte-Pair Encoding tokenizer built from scratch that can be trained on any raw text file. The `Tokenizer` class handles converting raw text into integer token IDs and back.
  * **GPT Model Architecture:** A standard GPT-2 style, decoder-only transformer model. The `GPTModel` class manages the token and positional embeddings, stacks multiple `TransformerLayer` instances, and includes a final linear layer to produce output logits over the vocabulary.
  * **HIP Acceleration:** All performance-critical operations, such as matrix multiplication, attention, and layer normalization, are implemented with custom HIP kernels for AMD GPUs.
  * **End-to-End Workflow:** The project includes scripts for downloading data, training the model from scratch, and generating new text.
  * **Self-Contained Build System:** The build process is managed by CMake, which automatically fetches the required JSON library.

### Documentation

- 🚀 [Getting Started](/getting-started/)  
- 🎓 [Training](/training/)  
- ✨ [Inference](/inference/)  
- 🏗 [Architecture](/architecture/)
- 💾 [Codebase](/codebase/)  

### Project Structure

For a deeper look into the codebase, you can explore the project structure:

  * `include/`: Contains all public header files for the model, kernels, and tokenizer.
  * `src/`: Contains the corresponding C++ source files for the project's components and executables.
  * `scripts/`: Holds helper scripts for downloading data and running the training process.