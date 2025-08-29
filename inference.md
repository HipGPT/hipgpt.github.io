---
layout: base
title: Inference
permalink: /inference/ 
nav_order: 2
---
# Inference

This section provides instructions and details on how to use a trained GPT model to generate new text. Inference, or text generation, is handled by the `generate` executable.

## 1. How Text Generation Works

Text generation is an iterative process. The model starts with a prompt and predicts the most likely next token based on the input sequence. This newly generated token is then added to the sequence, and the process repeats. This continues until the desired number of new tokens (`--num_tokens`) is reached or an end-of-sequence token (`--eos_id`) is generated.

The `generate` executable loads the trained model configuration and weights from a training run directory. It automatically resolves the tokenizer and checkpoint paths based on the run configuration. The `GPTModel::generate` method handles the core generation loop, which includes:

* Calling the forward pass of the model to get logits for the next token
* Applying repetition penalty to reduce repeated tokens
* Temperature scaling for controlling randomness
* Top_k and top_p (nucleus) filtering for improved sampling quality
* Sampling the next token from the filtered probability distribution
* Appending the new token to the sequence

For efficiency, the `generate` executable only feeds the most recent tokens up to `max_seq_len` into the model's forward pass during each step, using a sliding window approach.

## 2. Generating Text

### Using Run-Based Configuration

The recommended way to generate text is using the new run-based system with automatic configuration loading:

```bash
# Generate using the latest checkpoint from a training run
./build/generate --prompt "To be, or not to be:" --run-name shakespeare_v1
```

You can also specify a particular training step:

```bash
# Use a specific checkpoint step
./build/generate --prompt "Once upon a time" --run-name shakespeare_v1 --step 1500
```

### Advanced Generation Examples

Control the creativity and quality of generation with sampling parameters:

```bash
# Conservative, focused generation
./build/generate --prompt "The meaning of life is" \
  --run-name philosophy_model \
  --num_tokens 100 \
  --temp 0.7 \
  --top_k 20 \
  --top_p 0.9 \
  --rep-penalty 1.1

# Creative, diverse generation  
./build/generate --prompt "In a galaxy far away" \
  --run-name scifi_model \
  --num_tokens 200 \
  --temp 1.2 \
  --top_k 50 \
  --top_p 0.95
```

### Streaming Output

Generated text is displayed token by token as it's produced, providing real-time feedback during generation.

## 3. Configuration Resolution

The `generate` executable automatically resolves file paths based on your training run:

### Automatic Path Resolution
* **Config File**: Uses `latest_config.json` by default, or `[run-name]_step[N]_config.json` if `--step` is specified
* **Checkpoint**: Automatically loads the corresponding `.bin` file referenced in the config
* **Tokenizer**: Uses the tokenizer path from the configuration file

### Run Directory Structure
```
checkpoints/[run-name]/
├── tokenizer.json                    # ← Auto-loaded
├── tokens.bin  
├── [run-name]_step1000.bin          # ← Model weights
├── [run-name]_step1000_config.json  # ← Configuration
├── latest_checkpoint.bin → [symlink]
└── latest_config.json → [symlink]   # ← Default config
```

## 4. Command-Line Flags Explained

The `generate` executable is used to produce new text from a trained model with advanced sampling capabilities.

### Required Parameters
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt` | `string` | *(required)* | The initial text sequence for the model to continue. Enclose prompts with spaces in quotation marks. |
| `--run-name` | `string` | *(required)* | Name of the training run to use for generation. Loads configuration from `checkpoints/[run-name]/`. |

### Run and Checkpoint Selection
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--step` | `int` | *(latest)* | Specific training step to load. If not specified, uses the latest available checkpoint. |

### Generation Parameters  
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num_tokens` | `int` | `50` | The number of new tokens to generate after the prompt. |
| `--max_seq_len` | `int` | `32` | The context window size used during generation. Should match training configuration. |
| `--eos_id` | `int` | `-1` | The end-of-sequence token ID. Generation will stop if this token is produced. Set to `-1` to disable. |

### Sampling Control
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--top_k` | `int` | `5` | Restricts sampling to the top k most likely tokens. Higher values increase diversity. Set to `0` to disable. |
| `--temp` | `float` | `1.0` | Sampling temperature. Lower values (0.7) make output more focused, higher values (1.3) increase creativity. |
| `--top_p` | `float` | `0.9` | Nucleus sampling threshold. Dynamically adjusts the number of tokens considered based on cumulative probability. |
| `--rep-penalty` | `float` | `1.1` | Repetition penalty applied to previously generated tokens. Values > 1.0 reduce repetition. |

## 5. Sampling Strategies

HipGPT implements multiple advanced sampling techniques that work together:

### Temperature Scaling
Controls the randomness of predictions:
* `temp < 1.0`: More deterministic, focused output
* `temp = 1.0`: Unmodified model probabilities  
* `temp > 1.0`: More random, creative output

### Top-k Sampling
Restricts consideration to the k most probable tokens:
* Prevents extremely unlikely tokens from being selected
* Higher k values allow more diversity

### Top_p (Nucleus) Sampling
Dynamically selects tokens based on cumulative probability:
* More adaptive than top-k for different contexts
* Maintains quality while allowing flexibility

### Repetition Penalty
Reduces repetitive output by penalizing recently used tokens:
* Applied before temperature scaling
* Values between 1.05-1.15 typically work well

## 6. Performance Optimizations

The generation pipeline includes several performance enhancements:

* **Efficient Memory Management**: Reuses GPU buffers across generation steps
* **Sliding Window Context**: Maintains fixed-size context window for long generations  
* **Host-Side Sampling**: CPU-based sampling reduces GPU-CPU transfers
* **Vectorized Operations**: Optimized probability computations

## 7. Example Workflows

### Quick Test Generation
```bash
# Fast test with a simple prompt
./build/generate --prompt "Hello world" --run-name test_run --num_tokens 20
```

### High-Quality Story Generation
```bash
# Focused narrative generation
./build/generate \
  --prompt "The old wizard looked into the crystal ball and saw" \
  --run-name fantasy_model \
  --num_tokens 150 \
  --temp 0.8 \
  --top_k 30 \
  --top_p 0.92 \
  --rep-penalty 1.08
```

### Creative Exploration
```bash
# High creativity for brainstorming
./build/generate \
  --prompt "Imagine a world where" \
  --run-name creative_model \
  --num_tokens 100 \
  --temp 1.3 \
  --top_k 100 \
  --top_p 0.95 \
  --rep-penalty 1.05
```