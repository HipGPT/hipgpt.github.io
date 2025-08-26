---
layout: base
title: Architecture
permalink: /architecture/
---

# Architecture Overview

## Project Structure

The project is organized into `src` and `include` directories for a clean separation of source and header files.


```
├── build/                 # Build files (created by CMake)
├── data/                  # Data files (e.g., data.txt)
├── scripts/               # Helper scripts
│   ├── download_data.sh
│   └── run_train.sh
├── include/               # All public header files (.h)
│   ├── gpt_model.h
│   ├── hip_kernels.h
│   ├── tokenizer.h
│   └── transformer_layer.h
│
├── src/                   # All source files (.cpp)
│   ├── generate.cpp
│   ├── gpt_model.cpp
│   ├── hip_kernels.cpp
│   ├── tokenizer.cpp
│   ├── train_gpt.cpp
│   └── transformer_layer.cpp
│
├── CMakeLists.txt         # Main build configuration
├── LICENSE                # Project's license
└── README.md              # Project documentation
```

## Core Components

### `GPTModel`

The `GPTModel` class is the central component of the project. It integrates all the different parts of the neural network and manages the overall computation.

**Key Responsibilities:**

* **Initialization:** Sets up the model's layers, including token and positional embeddings, transformer layers, and the final output projection.
* **Forward Pass:** Takes input token IDs and computes the logits (raw predictions) for the next token in the sequence.
* **Backward Pass:** Computes gradients for all model parameters and updates them using an Adam optimizer. This is essential for training.
* **Text Generation:** Generates new text by iteratively predicting the next token and feeding it back into the model.
* **Checkpointing:** Provides functionality to save and load the model's state.

### `TransformerLayer`

The `TransformerLayer` class implements a single block of the transformer architecture.

**Key Components:**

* **Multi-Head Self-Attention:** Allows the model to weigh the importance of different words in the input sequence.
* **Feed-Forward Network:** A simple neural network that processes the output of the attention mechanism.
* **Layer Normalization and Residual Connections:** Crucial for stabilizing the training of deep neural networks.

### `Tokenizer`

The `Tokenizer` is responsible for converting raw text into a format that the model can understand.

**Functionality:**

* **Training:** Implements a Byte-Pair Encoding (BPE) algorithm that learns to merge frequent pairs of characters into a single new token.
* **Encoding/Decoding:** Converts text to token IDs and vice-versa.
* **Serialization:** The trained tokenizer's vocabulary can be saved to and loaded from a file.

### HIP Kernels

The `hip_kernels` files contain the low-level HIP code that runs on the GPU.

**Key Kernels:**

* Matrix Multiplication
* Embedding Lookup
* Multi-Head Attention
* Softmax and Loss Calculation
* Adam Optimizer
