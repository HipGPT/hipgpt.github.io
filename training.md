---
layout: base
title: Training
permalink: /training/
---

# Training

The following section describes how to train your own GPT model from scratch using the provided scripts.

## Training the Model

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

### Command-Line Flags Explained

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