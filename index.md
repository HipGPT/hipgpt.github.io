---
layout: default
title: Home
permalink: /
---

# Welcome to HipGPT Documentation

This project provides a C++ implementation of a Generative Pre-trained Transformer (GPT) model, accelerated using AMD's Heterogeneous-compute Interface for Portability (HIP). The implementation includes components for model definition, training, text generation, and tokenization.

### Key Features:

* **GPT Model Architecture:** A complete GPT model with embedding layers, multiple transformer blocks, and an output projection layer.
* **HIP Acceleration:** Computationally intensive operations are offloaded to the GPU using HIP kernels for high performance.
* **Training and Generation:** Scripts to train the model from scratch on a given text corpus and generate new text based on a prompt.
* **Byte-Pair Encoding (BPE) Tokenizer:** A trainable tokenizer to convert text into a sequence of integers that the model can process.
* **Self-Contained Build System**: Uses CMake to manage dependencies and build the project, automatically fetching the required JSON library.

### Next Steps

* To get started with setting up and running the project, see the **[Getting Started Guide](./getting-started.html)**.
* To learn about the internal workings of the model, check out the **[Architecture Overview](./architecture.html)**.