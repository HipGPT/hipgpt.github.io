---
layout: base
title: Codebase
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
├── scripts/                  # Automation scripts
│   ├── download\_data.sh      # Dataset fetching
│   └── run\_train.sh          # End-to-end training
├── include/                  # Public headers
│   ├── gpt\_model.h           # Main model interface
│   ├── hip\_kernels.h         # GPU kernel declarations
│   ├── tokenizer.h           # BPE tokenizer interface
│   └── transformer\_layer.h   # Transformer block interface
├── src/                      # Implementation files
│   ├── generate.cpp          # Text generation CLI
│   ├── gpt\_model.cpp         # Model orchestration
│   ├── hip\_kernels.cpp       # GPU kernel implementations
│   ├── tokenizer.cpp         # BPE tokenizer logic
│   ├── train\_gpt.cpp         # Training CLI
│   └── transformer\_layer.cpp # Transformer block logic
├── CMakeLists.txt            # Build configuration
├── LICENSE                   # MIT License
└── README.md                 # Project documentation

````

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
````

---

### Transformer Layer

{: .d-inline-block }
Core
{: .label .label-green }

**Files:** `src/transformer_layer.cpp`, `include/transformer_layer.h`

Implements a single GPT-style transformer block containing the fundamental attention and feed-forward mechanisms.

**Architecture Components:**

* **Multi-Head Self-Attention:** Parallel attention heads with QKV projections
* **Feed-Forward Network:** Two-layer MLP with ReLU activation
* **Residual Connections:** Skip connections around attention and FFN
* **Layer Normalization:** Pre-normalization for stable training
* **Dropout:** Regularization during training phase

---

### GPT Model

{: .d-inline-block }
Architecture
{: .label .label-purple }

**Files:** `src/gpt_model.cpp`, `include/gpt_model.h`

The main model class that orchestrates the complete GPT architecture by combining embeddings, transformer layers, and output projections.

**Model Pipeline:**

1. **Token Embeddings:** Maps token IDs to dense vectors
2. **Positional Embeddings:** Adds position information
3. **Transformer Stack:** Sequential transformer layers
4. **Output Projection:** Linear layer to vocabulary logits
5. **Loss Computation:** Cross-entropy for training

---

### HIP Kernels

{: .d-inline-block }
Performance
{: .label .label-yellow }

**Files:** `src/hip_kernels.cpp`, `include/hip_kernels.h`

Custom GPU kernels implemented in AMD HIP, providing transparent and educational implementations of all neural network operations.

**Implemented Operations:**

| Kernel Category    | Operations                                               |
| :----------------- | :------------------------------------------------------- |
| **Linear Algebra** | Matrix multiplication, bias addition, transpose variants |
| **Attention**      | Multi-head attention, scaled dot-product                 |
| **Activations**    | ReLU (fwd/bwd), Softmax                                  |
| **Normalization**  | LayerNorm (fwd/bwd), dropout                             |
| **Embeddings**     | Token/positional lookup and gradients                    |
| **Training**       | Cross-entropy loss, accuracy computation                 |
| **Optimization**   | Adam updates, SGD updates                                |
| **Sampling**       | Top-k sampling with temperature                          |
| **Utilities**      | Mean pooling, add-in-place ops                           |

---

## Application Entry Points

### Training Pipeline

{: .d-inline-block }
CLI
{: .label .label-red }

**File:** `src/train_gpt.cpp`

Complete training workflow from raw text to trained model using step-based training.

**Command-line Interface:**

```bash
./build/train_gpt \
  --data-path data/data.txt \
  --vocab-size 5000 \
  --seq 32 \
  --dim 128 \
  --heads 4 \
  --ff 256 \
  --layers 2 \
  --batch 4 \
  --steps 1000 \
  --lr 1e-2
```

**Available Parameters:**

* `--data-path`: Training text file (default: `data/data.txt`)
* `--tokenizer-path`: Tokenizer save path (default: `tokenizer.json`)
* `--tokens-path`: Tokenized dataset path (default: `tokens.bin`)
* `--vocab-size`: Maximum vocabulary size (default: 5000)
* `--seq`: Sequence length (default: 32)
* `--dim`: Embedding dimension (default: 128)
* `--heads`: Number of attention heads (default: 4)
* `--ff`: Feed-forward hidden dimension (default: 256)
* `--layers`: Number of transformer layers (default: 2)
* `--batch`: Batch size (default: 4)
* `--steps`: Training steps (default: 10)
* `--lr`: Learning rate (default: 1e-2)
* `--log-every`: Log frequency (default: 50)
* `--ckpt-every`: Checkpoint frequency (default: 500)
* `--keep-last`: Number of checkpoints to keep (default: 5)
* `--reset`: Force retrain tokenizer and tokens

---

### Text Generation

{: .d-inline-block }
CLI
{: .label .label-red }

**File:** `src/generate.cpp`

Interactive text generation interface for trained models.

**Usage Example:**

```bash
./build/generate \
  --prompt "To be or not to be" \
  --ckpt gpt_checkpoint.bin \
  --tokenizer tokenizer.json \
  --num_tokens 100 \
  --max_seq_len 32 \
  --temperature 0.8 \
  --top_k 10
```

**Available Parameters:**

* `--prompt`: Input text to continue (required)
* `--ckpt`: Model checkpoint file (default: `gpt_checkpoint.bin`)
* `--tokenizer`: Tokenizer file (default: `tokenizer.json`)
* `--num_tokens`: Number of tokens to generate (default: 50)
* `--max_seq_len`: Maximum sequence length (default: 32)
* `--top_k`: Top-k sampling parameter (default: 5)
* `--temp`: Temperature for sampling (default: 1.0)
* `--stream`: Stream output token by token (default: true)
* `--delay_ms`: Delay between tokens when streaming (default: 0)
* `--eos_id`: End-of-sequence token ID (default: -1)

{: .highlight }

> **Constraint:** The generation architecture (embed\_dim=128, heads=4, ff=256, layers=2) must match training configuration.

---

## Build System & Scripts

### CMake Configuration

**File:** `CMakeLists.txt`

* Configures HIP compilation with `hipcc`
* Fetches nlohmann/json
* Sets up include paths and linking
* Supports Debug/Release

### Helper Scripts

**Data Management:**

```bash
./scripts/download_data.sh
```

**Training Automation:**

```bash
./scripts/run_train.sh [custom_args...]
```

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
   ./scripts/run_train.sh
   ```

4. **Generate Text:**

   ```bash
   ./build/generate --ckpt gpt_checkpoint.bin --prompt "Once upon a time"
   ```

---

## Training Process

HipGPT uses step-based training rather than epoch-based training:

1. **Tokenizer Training:** BPE vocabulary learned from raw text
2. **Dataset Preparation:** Text encoded into token sequences
3. **Model Initialization:** Transformer layers and embeddings created
4. **Training Loop:** Fixed number of optimization steps with circular data iteration
5. **Checkpointing:** Weights saved periodically and at completion

---

## Summary

HipGPT's architecture is modular:

* **Tokenizer:** Text ↔ token conversion
* **TransformerLayer:** Core attention & FFN
* **GPTModel:** Full architecture orchestration
* **HIP Kernels:** GPU ops from scratch
* **Training/Generation:** End-to-end workflows

**Educational Goal:** Full transparency from **raw text → tokens → embeddings → attention → logits → generated text**.
