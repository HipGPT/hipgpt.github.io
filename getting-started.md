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
cmake ..
```

Compile the project: This command builds the executables.

```bash
cmake --build .
```

After the build completes, you will find the `train_gpt` and `generate` executables inside the build directory.

---

## 4. How to Use

### Training the Model

To train the model from scratch, you can use the `run_train.sh` script from the project's root directory. This script is a convenient wrapper around the `train_gpt` executable.

```bash
# From the project's root directory
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

The script will produce two key artifacts in your project's root directory:

* `tokenizer.json`: The trained vocabulary and merges.
* `gpt_checkpoint.bin`: The binary file containing the trained model weights.

You can customize the training process by passing command-line arguments directly to the script.
For example, to train for more steps with a smaller learning rate:

```bash
./scripts/run_train.sh --steps 1000 --lr 1e-3
```

---

### Generating Text

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
