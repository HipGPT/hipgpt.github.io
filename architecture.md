---
layout: base
title: Architecture
permalink: /architecture/
---

# Architecture Overview

The HipGPT project is designed around a standard GPT-2 style, decoder-only transformer model. The architecture is modular, with core components dedicated to specific tasks: tokenization, model definition, and low-level GPU computation.

## Core Components

### `GPTModel`

The `GPTModel` class is the central component of the project. It integrates all the different parts of the neural network and manages the overall computation.

**Key Responsibilities:**

* **Initialization:** The constructor of `GPTModel` sets up all the model's layers, including the token embedding layer, positional embedding layer, a stack of `TransformerLayer` instances, and the final output projection layer.
* **Forward Pass:** The `forward` method takes input token IDs and computes the logits (raw predictions) for the next token in the sequence. It performs an embedding lookup, then passes the result sequentially through each `TransformerLayer` before applying the final output projection.
* **Backward Pass:** The `backward` method computes gradients for all model parameters. It performs a re-computation of the forward pass to obtain intermediate activations, then backpropagates the gradients through the output projection and each `TransformerLayer` in reverse order. Finally, it updates the weights and biases of all layers using the Adam optimizer.
* **Text Generation:** The `generate` method generates new text by iteratively predicting the next token and feeding it back into the model. It uses the `forward` method to get logits and then samples the next token using a top-k and temperature-based sampling approach.
* **Checkpointing:** The `save_checkpoint` and `load_checkpoint` methods allow for saving and restoring the model's state, including all weights and biases, to a binary file. This enables resuming training or using a pre-trained model for inference.

### `TransformerLayer`

The `TransformerLayer` class implements a single block of the transformer architecture.

**Key Components:**

* **Multi-Head Self-Attention:** This mechanism allows the model to weigh the importance of different words in the input sequence. It is implemented by a `self_attention_forward` method, which projects the input into query, key, and value vectors, performs the attention calculation, and projects the result back to the model's dimension.
* **Feed-Forward Network:** A simple neural network that processes the output of the attention mechanism to allow the model to learn more complex features. The `feed_forward_forward` method handles this computation.
* **Layer Normalization and Residual Connections:** These are crucial for stabilizing the training of deep neural networks. The `forward` method applies layer normalization before both the attention and feed-forward sub-layers and adds residual connections by adding the sub-layer output to its input.

### `Tokenizer`

The `Tokenizer` is responsible for converting raw text into a format that the model can understand, and vice-versa.

**Functionality:**

* **Training:** The `train_bpe` method implements a Byte-Pair Encoding (BPE) algorithm. It learns to merge frequent pairs of characters and sub-words into a single new token until the vocabulary reaches a specified size limit.
* **Encoding/Decoding:** The `encode` method converts a string of text into a vector of integer IDs, while the `decode` method performs the reverse operation.
* **Serialization:** The trained tokenizer's vocabulary can be saved to a `tokenizer.json` file and loaded back for consistent tokenization.

### HIP Kernels

The `hip_kernels.cpp` file contains the low-level HIP code that performs the core computations on the GPU. These kernels are optimized for parallel execution on AMD GPUs.

**Key Kernels:**

* **Matrix Multiplication (`matmul_kernel`)**: Performs standard matrix multiplication.
* **Embedding Lookup (`embedding_lookup_kernel`)**: Looks up token and positional embeddings and sums them to create the initial input to the transformer blocks.
* **Multi-Head Attention (`multihead_attention_kernel`)**: Implements the self-attention mechanism by calculating dot products, applying a scaling factor, and performing a weighted sum.
* **Softmax and Loss Calculation (`softmax_loss_kernel`)**: Computes the softmax activation and the cross-entropy loss in a single kernel for efficiency.
* **Adam Optimizer (`adam_update_kernel`)**: Updates model weights using the Adam optimization algorithm.

## Model Size

The number of trainable parameters depends on the vocabulary size (`V`) learned from your dataset, along with the transformer hyperparameters:

**Parameter formula (no weight tying):**

```
Total =
    V·E                          (token embeddings)
  + S·E                          (positional embeddings)
  + L·(4E² + 2E·F + F + 9E)      (per transformer layer: QKV, O, FF1/FF2, LayerNorms)
  + E·V + V                      (final projection + bias)
```

Where:
- `E` = embedding dimension  
- `L` = number of layers  
- `H` = number of attention heads (`E` must be divisible by `H`)  
- `F` = feed-forward hidden dimension  
- `V` = vocabulary size  
- `S` = maximum sequence length  

**Default configuration:**  
`E=128, L=2, H=4, F=256, V≈5000, S=32`  
➡️ **~1.55M trainable parameters**

*Memory footprint:*  
- FP32 weights ≈ 6.2 MB  
(excluding optimizer states)