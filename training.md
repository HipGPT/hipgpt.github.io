---
layout: base
title: Training
permalink: /training/
---

# Training

The following section describes the process of training your own GPT model from scratch, including an overview of the data preparation, the training loop, and a detailed list of all available command-line flags.

## 1\. Data and Tokenizer

The training process begins with preparing the dataset and the tokenizer.

  * The `train_gpt` executable first checks for the existence of `tokenizer.json` and `tokens.bin`.
  * If these files are not found, or if the `--reset` flag is used, it will train a new Byte-Pair Encoding (BPE) tokenizer on the text file specified by `--data-path` (default: `data/data.txt`).
  * The trained tokenizer is then saved to `tokenizer.json`, and the entire dataset is tokenized and saved as a binary file to `tokens.bin`. This binary file speeds up future training runs by skipping the tokenization step.
  * The final vocabulary size is determined by the `vocab-size` parameter.

The tokenizer is a custom Byte-Pair Encoding (BPE) tokenizer that can be trained from scratch on any raw text file. Its primary function is to convert raw text into a sequence of integer token IDs that the model can understand, and to convert these IDs back into human-readable text.

The tokenizer works as follows:

1.  **Initialization**: The tokenizer first builds an initial vocabulary from the individual characters in the training text. Special tokens, like `</w>` to denote the end of a word, are also added to the initial vocabulary.
2.  **Training**: The tokenizer's `train_bpe` method is responsible for learning the vocabulary from a given text. It repeatedly finds the most frequent pair of tokens in the corpus and merges them into a new, single token. This process continues until the vocabulary reaches a predefined size limit (`vocab_limit_`), which can be set using the `--vocab-size` flag during training.
3.  **Encoding**: The `encode` method takes a string of text and converts it into a vector of integer IDs. To do this, it first splits the text into words, then for each word, it applies the merges learned during training to break the word down into the largest possible sub-word tokens from the vocabulary. A cache is used to speed up the encoding of repeated words. If a token is not in the vocabulary, it is silently dropped.
4.  **Decoding**: The `decode` method performs the reverse operation, converting a vector of integer IDs back into a human-readable string. It removes the special end-of-word token `</w>` and adds a space to reconstruct the original text structure.

## 2\. Training the Model

To train the model from scratch, you can use the `run_train.sh` script from the project's root directory. This script is a convenient wrapper around the `build/train_gpt` executable.

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

During training, the model's loss and accuracy are printed at intervals defined by `--log-every`. The script can also save periodic checkpoints to a file named `gpt_checkpoint_step[step_number].bin` at intervals defined by `--ckpt-every`. The `prune_old_checkpoints` function is used to keep only the most recent checkpoints, preventing the accumulation of large files.

### 3\. Command-Line Flags Explained

The `run_train.sh` script is a wrapper for the `build/train_gpt` executable. The following flags can be passed to the script to customize the training process.

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