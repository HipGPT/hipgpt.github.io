---
layout: base
title: Codebase
permalink: /codebase/
nav_order: 3
---

# Codebase Overview
{: .no_toc }

This page provides a comprehensive guide to the HipGPT codebase architecture and explains the purpose of each component. The project is designed to be **self-contained**, **readable**, and **educational**, allowing you to understand how a GPT-style transformer works from the ground up.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Project Structure

```
hipgpt/
├── build/                    # Build outputs (CMake artifacts)
├── data/                     # Training datasets
│   └── data.txt
├── checkpoints/              # Training runs and model checkpoints
│   └── [run-name]/
│       ├── tokenizer.json
│       ├── tokens.bin
│       ├── [run-name]_stepN.bin
│       ├── [run-name]_stepN_config.json
│       ├── latest_checkpoint.bin → [symlink]
│       └── latest_config.json → [symlink]
├── scripts/                  # Automation scripts
│   ├── download_data.sh      # Dataset fetching
│   ├── run_train.sh          # End-to-end training
│   └── run_generate.sh       # Text generation wrapper
├── include/                  # Public headers
│   ├── gpt_model.h           # Main model interface
│   ├── hip_kernels.h         # GPU kernel declarations
│   ├── tokenizer.h           # BPE tokenizer interface
│   └── transformer_layer.h   # Transformer block interface
├── src/                      # Implementation files
│   ├── generate.cpp          # Text generation CLI
│   ├── gpt_model.cpp         # Model orchestration
│   ├── hip_kernels.cpp       # GPU kernel implementations
│   ├── tokenizer.cpp         # BPE tokenizer logic
│   ├── train_gpt.cpp         # Training CLI
│   └── transformer_layer.cpp # Transformer block logic
├── CMakeLists.txt            # Build configuration
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Core Components

### Tokenizer
{: .d-inline-block }
Essential
{: .label .label-blue }

**Files:** `src/tokenizer.cpp`, `include/tokenizer.h`

The tokenizer implements Byte-Pair Encoding (BPE) from scratch, handling the conversion between human-readable text and model-compatible token sequences.

**Key Responsibilities:**
- **Vocabulary Training:** Learns subword tokens from training corpus
- **Text Encoding:** Converts strings to token ID sequences  
- **Text Decoding:** Reconstructs text from token sequences
- **Serialization:** Saves/loads trained vocabulary using JSON

**Usage Example:**
```cpp
Tokenizer tokenizer(5000);
tokenizer.train_bpe(text);
tokenizer.save("tokenizer.json");
auto tokens = tokenizer.encode("Hello, world!");
std::string text = tokenizer.decode(tokens);
```

---

### Transformer Layer
{: .d-inline-block }
Core
{: .label .label-green }

**Files:** `src/transformer_layer.cpp`, `include/transformer_layer.h`

Implements a single GPT-style transformer block containing the fundamental attention and feed-forward mechanisms with advanced optimizations.

**Architecture Components:**

* **Multi-Head Self-Attention:** Parallel attention heads with QKV projections and Flash Attention optimization
* **Feed-Forward Network:** Two-layer MLP with ReLU activation
* **Residual Connections:** Skip connections around attention and FFN
* **Layer Normalization:** Pre-normalization for stable training with efficient GPU reductions
* **Dropout:** Configurable regularization during training phase
* **Adam Optimization:** Built-in Adam optimizer states for all parameters

**Advanced Features:**
* **Memory Management:** Dynamic buffer allocation based on batch size
* **Gradient Accumulation:** Proper backpropagation through all components
* **Parameter Management:** Comprehensive save/load for all weights and optimizer states

---

### GPT Model
{: .d-inline-block }
Architecture
{: .label .label-purple }

**Files:** `src/gpt_model.cpp`, `include/gpt_model.h`

The main model class that orchestrates the complete GPT architecture by combining embeddings, transformer layers, and output projections with advanced training features.

**Model Pipeline:**

1. **Token Embeddings:** Maps token IDs to dense vectors
2. **Positional Embeddings:** Adds position information
3. **Transformer Stack:** Sequential transformer layers with residual connections
4. **Output Projection:** Linear layer to vocabulary logits
5. **Loss Computation:** Cross-entropy with numerical stability improvements

**Training Features:**
* **Gradient Clipping:** L2 norm clipping to prevent gradient explosion
* **Adam Optimization:** Advanced optimizer with momentum and bias correction
* **Checkpoint Management:** Complete model state serialization

**Generation Features:**
* **Advanced Sampling:** Top-k, top-p (nucleus), temperature, and repetition penalty
* **Sliding Window:** Efficient long sequence generation with context management
* **Memory Efficiency:** Optimized GPU memory allocation during inference

---

### HIP Kernels
{: .d-inline-block }
Performance
{: .label .label-yellow }

**Files:** `src/hip_kernels.cpp`, `include/hip_kernels.h`

Custom GPU kernels implemented in AMD HIP, providing transparent and educational implementations of all neural network operations with significant performance optimizations.

**Implemented Operations:**

| Kernel Category    | Operations                                               |
| :----------------- | :------------------------------------------------------- |
| **Linear Algebra** | Tiled matrix multiplication with shared memory, bias addition, transpose variants |
| **Attention**      | Flash Attention implementation, scaled dot-product with numerical stability |
| **Activations**    | ReLU (fwd/bwd), Softmax with improved numerical stability |
| **Normalization**  | LayerNorm with efficient block reductions, dropout with device-side RNG |
| **Embeddings**     | Vectorized token/positional lookup, gradient accumulation |
| **Training**       | Cross-entropy loss with stability, accuracy computation |
| **Optimization**   | Adam updates with bias correction, gradient L2 norm computation |
| **Sampling**       | Advanced top-k/top-p sampling with temperature scaling |
| **Memory Utilities** | Gradient clipping, in-place operations, mean pooling |

**Performance Optimizations:**
* **Tiled Matrix Multiplication:** Shared memory usage for improved cache efficiency
* **Flash Attention:** Memory-efficient attention for supported head dimensions (32, 64)  
* **Vectorized Memory Access:** float4 operations for improved bandwidth utilization
* **Efficient Reductions:** Warp-level and block-level parallel reductions
* **Numerical Stability:** Improved softmax, layer normalization, and loss computations

---

## Application Entry Points

### Training Pipeline
{: .d-inline-block }
CLI
{: .label .label-red }

**File:** `src/train_gpt.cpp`

Complete training workflow from raw text to trained model using step-based training with comprehensive checkpoint management.

**Run-Based Training System:**
```bash
./build/train_gpt \
  --data-path data/data.txt \
  --run-name shakespeare_v1 \
  --vocab-size 5000 \
  --seq 32 \
  --dim 128 \
  --heads 4 \
  --ff 256 \
  --layers 2 \
  --batch 4 \
  --steps 2000 \
  --lr 1e-2 \
  --ckpt-every 500 \
  --keep-last 5
```

**Key Features:**
* **Run Organization:** Each training session stored in `checkpoints/[run-name]/`
* **Checkpoint Pruning:** Automatic cleanup of old checkpoints
* **Resume Training:** Seamless continuation from any checkpoint
* **Configuration Management:** JSON configs saved with each checkpoint
* **Progress Tracking:** Loss, perplexity, accuracy, and timing metrics

---

### Text Generation
{: .d-inline-block }
CLI
{: .label .label-red }

**File:** `src/generate.cpp`

Interactive text generation interface for trained models with advanced sampling capabilities.

**Run-Based Generation:**
```bash
./build/generate \
  --prompt "To be or not to be" \
  --run-name shakespeare_v1 \
  --step 2000 \
  --num_tokens 100 \
  --temp 0.8 \
  --top_k 20 \
  --top_p 0.9 \
  --rep-penalty 1.1
```

**Advanced Features:**
* **Automatic Configuration:** Loads model config from run directory
* **Multiple Sampling Methods:** Top-k, top-p, temperature, repetition penalty
* **Streaming Output:** Real-time token generation display
* **Context Management:** Sliding window for long sequence generation

---

## Build System & Scripts

### CMake Configuration
**File:** `CMakeLists.txt`

* Configures HIP compilation with `hipcc`
* Fetches nlohmann/json dependency
* Sets up include paths and linking
* Supports Debug/Release builds

### Helper Scripts

**Data Management:**
```bash
./scripts/download_data.sh
```

**Training Automation:**
```bash
./scripts/run_train.sh [custom_args...]
```

**Generation Wrapper:**
```bash
./scripts/run_generate.sh --prompt "Hello" --run-name my_model
```

---

## Advanced Features

### Flash Attention Implementation
HipGPT includes an optimized Flash Attention implementation that provides memory-efficient attention computation:

* **Supported Configurations:** Head dimensions of 32 and 64 with sequences up to 512
* **Memory Efficiency:** Processes attention in blocks to fit in GPU shared memory
* **Numerical Stability:** Online softmax computation with stable exponentials
* **Fallback Support:** Automatic fallback to standard attention for unsupported configurations

### Gradient Clipping and Optimization
The training pipeline includes robust gradient management:

* **Device-Side Clipping:** L2 norm computation and scaling entirely on GPU
* **Global Gradient Norm:** Accumulates norms across all parameter groups
* **Configurable Threshold:** Default maximum norm of 1.0 (adjustable)
* **Adam with Bias Correction:** Proper bias correction for momentum terms

### Memory Management
Comprehensive memory management for efficient GPU utilization:

* **Dynamic Buffer Allocation:** Temporary buffers allocated based on batch size
* **Automatic Cleanup:** RAII-style memory management in destructors
* **Memory Reuse:** Buffers reused across training steps and generation sequences
* **Error Handling:** Proper error checking for all GPU allocations

---

## Getting Started

1. **Clone and Build:**
   ```bash
   git clone https://github.com/aarnetalman/HipGPT.git
   cd HipGPT
   mkdir build && cd build
   cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
   make
   ```

2. **Download Data:**
   ```bash
   ./scripts/download_data.sh
   ```

3. **Train Model:**
   ```bash
   ./scripts/run_train.sh --run-name my_first_model --steps 1000
   ```

4. **Generate Text:**
   ```bash
   ./scripts/run_generate.sh --prompt "Once upon a time" --run-name my_first_model
   ```

---

## Training Process

HipGPT uses step-based training with comprehensive checkpoint management:

1. **Run Initialization:** Creates organized directory structure for each training session
2. **Tokenizer Training:** BPE vocabulary learned from raw text and cached
3. **Dataset Preparation:** Text encoded into token sequences and stored efficiently
4. **Model Initialization:** Transformer layers and embeddings created with proper weight initialization
5. **Training Loop:** Fixed number of optimization steps with circular data iteration
6. **Advanced Optimization:** Gradient clipping, Adam updates, and numerical stability improvements
7. **Checkpoint Management:** Automatic saving, pruning, and symlink management

### Key Training Improvements

**Gradient Clipping Pipeline:**
* Device-side L2 norm computation across all parameter groups
* Automatic scaling when gradients exceed threshold (default: 1.0)
* Applied before optimizer updates for training stability

**Memory Optimization:**
* Dynamic buffer allocation based on actual batch sizes
* Proper cleanup and memory reuse patterns
* RAII-style resource management

**Numerical Stability:**
* Improved softmax computation with better overflow handling
* Enhanced layer normalization with efficient parallel reductions
* Stable cross-entropy loss computation

---

## Advanced Kernel Implementations

### Flash Attention
The HIP kernels include a custom Flash Attention implementation:

```cpp
template<int HEAD_DIM>
__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* output,
    int B, int S, int E, int H
)
```

**Features:**
* **Memory Efficient:** Processes attention in blocks to fit in shared memory
* **Online Softmax:** Stable computation without storing full attention matrix
* **Template Specialization:** Optimized versions for head dimensions 32 and 64
* **Fallback Support:** Automatic fallback for unsupported configurations

### Tiled Matrix Multiplication
Optimized matrix operations with shared memory tiling:

```cpp
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K)
```

**Optimizations:**
* **Shared Memory Tiling:** 32x32 tiles for improved cache utilization
* **Memory Coalescing:** Optimized memory access patterns
* **Bank Conflict Avoidance:** Careful shared memory layout design

### Layer Normalization
Efficient layer normalization with parallel reductions:

```cpp
__global__ void layer_norm_forward(
    const float* input, float* output, 
    const float* gamma, const float* beta,
    int N, int E, float eps = 1e-5f
)
```

**Features:**
* **Warp-Level Reductions:** Fast parallel computation of mean and variance
* **Vectorized Memory Access:** float4 operations where possible
* **Numerical Stability:** Robust handling of small variances

---

## Configuration Management

### JSON Configuration System
Training runs generate comprehensive configuration files:

```json
{
  "model": {
    "vocab_size": 5000,
    "max_seq_len": 32,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_hidden_dim": 256,
    "num_layers": 2
  },
  "tokenizer": {
    "path": "tokenizer.json",
    "tokens_path": "tokens.bin"
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 0.01,
    "steps": 2000
  },
  "checkpoint": {
    "latest": "model_step2000.bin",
    "step": 2000
  }
}
```

### Automatic Path Resolution
The generation pipeline automatically resolves file paths:
* Configuration files specify relative paths within run directories
* Symlinks maintain references to latest checkpoints
* Error handling for missing or corrupted configurations

---

## Summary

HipGPT's architecture is modular and optimized:

* **Tokenizer:** Efficient BPE implementation with caching
* **TransformerLayer:** Advanced attention and FFN with Flash Attention
* **GPTModel:** Complete architecture with gradient clipping and Adam optimization
* **HIP Kernels:** High-performance GPU operations with memory optimizations
* **Training/Generation:** Production-ready workflows with comprehensive checkpoint management

**Educational Goal:** Full transparency from **raw text → tokens → embeddings → attention → logits → generated text** with industry-standard optimizations and numerical stability improvements.