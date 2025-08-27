---
layout: base
title: Inference
permalink: /inference/ 
---
# Inference

This section provides instructions and details on how to use a trained GPT model to generate new text. Inference, or text generation, is handled by the `generate` executable.

## 1\. How Text Generation Works

Text generation is an iterative process. The model starts with a prompt and predicts the most likely next token based on the input sequence. This newly generated token is then added to the sequence, and the process repeats. This continues until the desired number of new tokens (`--num_tokens`) is reached or an end-of-sequence token (`--eos_id`) is generated.

The `generate` executable first loads the trained `Tokenizer` from `tokenizer.json` and the `GPTModel` weights from `gpt_checkpoint.bin`. It then tokenizes the user-provided prompt and prepares the input for the model. The `GPTModel::generate` method handles the core generation loop, which includes:

  * Calling the forward pass of the model to get logits for the next token.
  * Sampling the next token from these logits based on specified parameters like `top_k` and `temperature`.
  * Appending the new token to the sequence.

For efficiency, the `generate` executable only feeds the most recent tokens up to `max_seq_len` into the model's forward pass during each step.

## 2\. Generating Text

Once the model is trained and you have a `gpt_checkpoint.bin` file, you can use the `generate` executable to create new text based on a prompt.

```bash
# Run from the build directory
./build/generate --prompt "To be, or not to be:"
```

You can control the output with several parameters, as shown in this more advanced example:

```bash
./build/generate --prompt "My kingdom for a" --num_tokens 100 --top_k 50 --temp 0.8
```

Output can also be streamed token by token, with an optional delay, by using the `--stream` and `--delay_ms` flags.

### 3\. Command-Line Flags Explained

The `generate` executable is used to produce new text from a trained model.

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt` | `string` | *(required)* | The initial text sequence for the model to continue. Enclose prompts with spaces in quotation marks. |
| `--num_tokens` | `int` | `50` | The number of new tokens to generate after the prompt. |
| `--max_seq_len` | `int` | `32` | The context window size used during generation. This should match the `--seq` length used during training. |
| `--ckpt` | `string` | `"gpt_checkpoint.bin"` | Path to the binary file containing the trained model weights. |
| `--tokenizer` | `string` | `"tokenizer.json"` | Path to the trained tokenizer file. |
| `--top_k` | `int` | `5` | Restricts the sampling to the top k most likely tokens. This helps prevent incoherent output. Set to `0` to disable. |
| `--temp` | `float` | `1.0` | Sampling temperature. A lower value (e.g., 0.7) makes the output more deterministic and focused, while a higher value (e.g., 1.5) increases randomness and creativity. |
| `--eos_id` | `int` | `-1` | The end-of-sequence token ID. Generation will stop if this token is produced. Set to `-1` to disable. |
| `--stream` | `bool` | `true` | If `true`, output is streamed token by token. If `false`, the full generated text is printed at once. |
| `--delay_ms` | `int` | `0` | Delay in milliseconds between printing each token when `--stream true` is used. |