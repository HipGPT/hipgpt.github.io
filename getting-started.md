---
layout: base
title: Getting Started
permalink: /getting-started/
---

# Getting Started

Follow these steps to download the dataset, build the executables, and start using the model.

## 1. Prerequisites

* An **AMD GPU** compatible with the ROCm toolkit.
* The **ROCm Toolkit** (version 5.0 or newer) installed.
* **CMake** (version 3.21 or newer).
* A C++ compiler (like `g++` or `clang++`).
* `git` and `wget` for downloading dependencies and data.

---

## 2. Download the Dataset

The project includes a convenient script to download the Tiny Shakespeare dataset, which is a great starting point for training.

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
````

This command will create a `data/` directory and place a `data.txt` file inside it.

---

## 3. Build the Project

The project uses CMake to handle the entire build process, including fetching dependencies.

Create a build directory: It's best practice to build the project in a separate directory to keep the main folder clean.

```bash
mkdir build
cd build
```

Run CMake to configure the project: This step finds the necessary tools and prepares the build files.

```bash
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
```

Compile the project: This command builds the executables.

```bash
make
```

After the build completes, you will find the `train_gpt` and `generate` executables inside the build directory.

---

### 4\. How to Use

#### Training the Model

To train the model from scratch, you can use the `run_train.sh` script from the project's root directory. This script is a convenient wrapper around the `train_gpt` executable.

```bash
# From the project's root directory
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

The script will produce two key artifacts in your project's root directory:

  * `tokenizer.json`: The trained vocabulary and merges.
  * `gpt_checkpoint.bin`: The binary file containing the trained model weights.

You can customize the training process by passing command-line arguments directly to the script. For example, to train for more steps with a smaller learning rate:

```bash
./scripts/run_train.sh --steps 1000 --lr 1e-3
```

#### Generating Text

Once the model is trained and you have a `gpt_checkpoint.bin` file, you can use the `generate` executable to create new text based on a prompt.

```bash
# Run from the build directory
./generate --prompt "To be, or not to be:"
```

You can control the output with several parameters:

  * `--prompt "<text>"`: The initial text to start generation from. (**Required**)
  * `--num_tokens N`: The number of new tokens to generate (default: 50).
  * `--top_k N`: Restricts sampling to the top k most likely tokens, which can improve quality (default: 5).
  * `--temp F`: Controls the randomness of the output. Higher values (e.g., 1.0) are more random, while lower values (e.g., 0.7) are more deterministic (default: 1.0).

Here is a more advanced example:

```bash
./generate --prompt "My kingdom for a" --num_tokens 100 --top_k 50 --temp 0.8
```

-----

### Command-Line Flags Explained

#### Training (`scripts/run_train.sh`)

The `run_train.sh` script is a wrapper for the `build/train_gpt` executable. The following flags can be passed to the script to customize the training process. The C++ code parses these flags to set the training parameters.

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-path` | `string` | `"data/data.txt"` | Path to the dataset text file to be used for training. |
| `--tokenizer-path` | `string` | `"tokenizer.json"` | Path to save/load the trained tokenizer. |
| `--tokens-path` | `string` | `"tokens.bin"` | Path to save/load the pre-tokenized dataset as a binary file. This speeds up subsequent runs. |
| `--reset` | `flag` | *(none)* | If present, forces the program to retrain the tokenizer and re-tokenize the dataset, even if `tokenizer.json` and `tokens.bin` already exist. |
| `--seq` | `int` | `32` | The maximum length of a training sequence (context window). |
| `--batch` | `int` | `4` | The number of sequences per training batch. |
| `--steps` | `int` | `10` | The total number of training steps (iterations). |
| `--lr` | `float` | `1e-2` | The learning rate for the AdamW optimizer. |
| `--log-every` | `int` | `50` | Frequency (in steps) to print training progress and loss. |
| `--ckpt-every` | `int` | `500` | Frequency (in steps) to save a model checkpoint. Set to `0` to disable periodic checkpoints. |
| `--keep-last` | `int` | `5` | The number of recent periodic checkpoints to keep. Older ones will be pruned. Only relevant if `--ckpt-every` is greater than 0. |
| `--vocab-size` | `int` | `5000` | The maximum size of the vocabulary to be created by the BPE tokenizer. |
| `--dim` | `int` | `128` | The embedding dimension (also the model dimension). |
| `--heads` | `int` | `4` | The number of attention heads in the Multi-Head Attention layers. |
| `--ff` | `int` | `256` | The hidden dimension of the feed-forward network within each transformer block. |
| `--layers` | `int` | `2` | The number of transformer layers in the model. |

#### Generation (`build/generate`)

The `generate` executable is used to produce new text from a trained model.

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt` | `string` | *(required)* | The initial text sequence for the model to continue. Enclose prompts with spaces in quotation marks. |
| `--num_tokens` | `int` | `50` | The number of new tokens to generate after the prompt. |
| `--max_seq_len` | `int` | `32` | The context window size used during generation. This should match the `block_size` used during training. |
| `--ckpt` | `string` | `"gpt_checkpoint.bin"` | Path to the binary file containing the trained model weights. |
| `--tokenizer` | `string` | `"tokenizer.json"` | Path to the trained tokenizer file. |
| `--top_k` | `int` | `5` | Restricts the sampling to the top k most likely tokens. This helps prevent incoherent output. Set to `0` to disable. |
| `--temp` | `float` | `1.0` | Sampling temperature. A lower value (e.g., 0.7) makes the output more deterministic and focused, while a higher value (e.g., 1.5) increases randomness and creativity. |
| `--eos_id` | `int` | `-1` | The end-of-sequence token ID. Generation will stop if this token is produced. Set to `-1` to disable. |
| `--stream` | `bool` | `true` | If `true`, output is streamed token by token. If `false`, the full generated text is printed at once. |
| `--delay_ms` | `int` | `0` | Delay in milliseconds between printing each token when `--stream true` is used. |