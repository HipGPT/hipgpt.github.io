---
layout: base
title: Home
permalink: /
---

# Welcome to HipGPT Documentation 🚀

This project provides a lightweight, from-scratch implementation of a GPT-2 style transformer model written in C++ and accelerated using AMD's **[HIP API](https://rocm.docs.amd.com/en/latest/understand/hip_api/hip_api.html)** for ROCm-enabled GPUs. The project is designed to be a clear and understandable guide to the inner workings of large language models.

The code is intended for educational purposes and is not recommended for production use of any kind. 

### Key Features:

  * **Custom BPE Tokenizer:** A Byte-Pair Encoding tokenizer built from scratch that can be trained on any raw text file. The `Tokenizer` class handles converting raw text into integer token IDs and back.
  * **GPT Model Architecture:** A standard GPT-2 style, decoder-only transformer model. The `GPTModel` class manages the token and positional embeddings, stacks multiple `TransformerLayer` instances, and includes a final linear layer to produce output logits over the vocabulary.
  * **HIP Acceleration:** All performance-critical operations, such as matrix multiplication, attention, and layer normalization, are implemented with custom HIP kernels for AMD GPUs.
  * **End-to-End Workflow:** The project includes scripts for downloading data, training the model from scratch, and generating new text.
  * **Self-Contained Build System:** The build process is managed by CMake, which automatically fetches the required JSON library.

### Getting Started

To get started with HipGPT, follow the instructions on the [Getting Started](/getting-started/) page. You will learn how to set up the prerequisites, clone the repository, download the dataset, and build the project executables.

### How to Use

  * **Training:** The [Training](/training/) page provides detailed information on how to train the model, including a list of all available command-line flags for customizing the training process.
  * **Inference:** Visit the [Inference](/inference/) page to learn how to use a trained model to generate new text. This page also explains all the command-line options for controlling the text generation.

### Project Structure

For a deeper look into the codebase, you can explore the project structure:

  * `include/`: Contains all public header files for the model, kernels, and tokenizer.
  * `src/`: Contains the corresponding C++ source files for the project's components and executables.
  * `scripts/`: Holds helper scripts for downloading data and running the training process.