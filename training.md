---
layout: base
title: Training
permalink: /training/
nav_order: 1
---

# Training

The following section describes the process of training your own GPT model from scratch, including an overview of the data preparation, the training loop, and a detailed list of all available command-line flags.

## 1. Data and Tokenizer

The training process begins with preparing the dataset and the tokenizer.

* The `train_gpt` executable first checks for the existence of `tokenizer.json` and `tokens.bin` in the run directory.
* If these files are not found, or if the `--reset` flag is used, it will train a new Byte-Pair Encoding (BPE) tokenizer on the text file specified by `--data-path` (default: `data/data.txt`).
* The trained tokenizer is then saved to the run directory as `tokenizer.json`, and the entire dataset is tokenized and saved as a binary file to `tokens.bin`. This binary file speeds up future training runs by skipping the tokenization step.
* The final vocabulary size is determined by the `vocab-size` parameter.

The tokenizer is a custom Byte-Pair Encoding (BPE) tokenizer that can be trained from scratch on any raw text file. Its primary function is to convert raw text into a sequence of integer token IDs that the model can understand, and to convert these IDs back into human-readable text.

The tokenizer works as follows:

1. **Initialization**: The tokenizer first builds an initial vocabulary from the individual characters in the training text. Special tokens, like `</w>` to denote the end of a word, are also added to the initial vocabulary.
2. **Training**: The tokenizer's `train_bpe` method is responsible for learning the vocabulary from a given text. It repeatedly finds the most frequent pair of tokens in the corpus and merges them into a new, single token. This process continues until the vocabulary reaches a predefined size limit (`vocab_limit_`), which can be set using the `--vocab-size` flag during training.
3. **Encoding**: The `encode` method takes a string of text and converts it into a vector of integer IDs. To do this, it first splits the text into words, then for each word, it applies the merges learned during training to break the word down into the largest possible sub-word tokens from the vocabulary. A cache is used to speed up the encoding of repeated words. If a token is not in the vocabulary, it is silently dropped.
4. **Decoding**: The `decode` method performs the reverse operation, converting a vector of integer IDs back into a human-readable string. It removes the special end-of-word token `</w>` and adds a space to reconstruct the original text structure.

## 2. Training Runs and Checkpoint Management

HipGPT uses a **run-based training system** that organizes each training session with unique identifiers and comprehensive checkpoint management.

### Run Organization

Each training run is stored in a separate directory:
```
checkpoints/
├── run-name/
│   ├── tokenizer.json
│   ├── tokens.bin
│   ├── [run-name]_step100.bin
│   ├── [run-name]_step100_config.json
│   ├── latest_checkpoint.bin → [symlink]
│   └── latest_config.json → [symlink]
```

### Checkpoint Features

* **Automatic Checkpointing**: Saves model weights and configuration at specified intervals (`--ckpt-every`)
* **Checkpoint Pruning**: Keeps only the most recent checkpoints (`--keep-last`) to manage disk space
* **Resume Training**: Can resume from any checkpoint using `--ckpt` flag
* **Symlink Management**: Maintains `latest_checkpoint.bin` and `latest_config.json` symlinks for easy access

## 3. Training the Model

To train the model from scratch, you can use the `run_train.sh` script from the project's root directory. This script is a convenient wrapper around the `build/train_gpt` executable.

```bash
# From the project's root directory
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

### Named Training Runs

You can create organized training runs with custom names:

```bash
./scripts/run_train.sh --run-name shakespeare_v1 --steps 2000
```

This creates a `checkpoints/shakespeare_v1/` directory with all associated files.

### Resuming Training

To continue training from a checkpoint:

```bash
./scripts/run_train.sh --run-name shakespeare_v1 --ckpt checkpoints/shakespeare_v1/shakespeare_v1_step1000.bin --steps 1000
```

The script will produce organized artifacts in the run directory:

* `tokenizer.json`: The trained vocabulary and merges
* `tokens.bin`: Pre-tokenized dataset for efficient loading
* `[run-name]_stepN.bin`: Model weights at step N
* `[run-name]_stepN_config.json`: Complete configuration for step N
* `latest_checkpoint.bin` and `latest_config.json`: Symlinks to most recent files

During training, the model's loss, perplexity, accuracy, and timing are printed at intervals defined by `--log-every`. The script automatically saves periodic checkpoints and prunes old ones to prevent excessive disk usage.

## 4. Command-Line Flags Explained

The `run_train.sh` script is a wrapper for the `build/train_gpt` executable. The following flags can be passed to the script to customize the training process.

### Data and Tokenization

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-path` | `string` | `"data/data.txt"` | Path to the dataset text file to be used for training. |
| `--vocab-size` | `int` | `5000` | The maximum size of the vocabulary to be created by the BPE tokenizer. |
| `--reset` | `flag` | *(none)* | If present, forces the program to retrain the tokenizer and re-tokenize the dataset, even if files already exist in the run directory. |

### Model Architecture

| Flag       | Type  | Default | Description |
|------------|-------|---------|-------------|
| `--dim`    | int   | 256     | The embedding dimension (also the model dimension). |
| `--heads`  | int   | 8       | The number of attention heads in the Multi-Head Attention layers. |
| `--ff`     | int   | 1024    | The hidden dimension of the feed-forward network within each transformer block. |
| `--layers` | int   | 8       | The number of transformer layers in the model. |
| `--seq`    | int   | 256     | The maximum length of a training sequence (context window). |

### Training Configuration

| Flag      | Type   | Default | Description |
|-----------|--------|---------|-------------|
| `--batch` | int    | 32      | The number of sequences per training batch. |
| `--steps` | int    | 50000   | The total number of training steps (iterations). |
| `--lr`    | float  | 3e-4    | The learning rate for the Adam optimizer. |

### Logging and Checkpointing

| Flag          | Type | Default | Description |
|---------------|------|---------|-------------|
| `--log-every` | int  | 50      | Frequency (in steps) to print training progress, loss, perplexity, and accuracy. |
| `--ckpt-every`| int  | 1000    | Frequency (in steps) to save a model checkpoint. Set to `0` to disable periodic checkpoints. |
| `--keep-last` | int  | 5       | The number of recent periodic checkpoints to keep. Older ones will be automatically pruned. |


## 5. Training Process Details

HipGPT uses step-based training rather than epoch-based training:

1. **Tokenizer Training**: BPE vocabulary learned from raw text
2. **Dataset Preparation**: Text encoded into token sequences and cached
3. **Model Initialization**: Transformer layers and embeddings created with random weights
4. **Training Loop**: Fixed number of optimization steps with circular data iteration
5. **Gradient Clipping**: L2 norm clipping applied to prevent gradient explosion
6. **Adam Optimization**: Advanced optimizer with momentum and adaptive learning rates
7. **Checkpointing**: Weights and configuration saved periodically and at completion

### Advanced Features

* **Gradient Clipping**: Automatic L2 norm clipping with configurable maximum norm (default: 1.0)
* **Flash Attention**: Optimized attention implementation for supported head dimensions (32, 64)
* **Memory Management**: Efficient GPU memory allocation with proper cleanup
* **Numerical Stability**: Improved softmax and layer normalization implementations