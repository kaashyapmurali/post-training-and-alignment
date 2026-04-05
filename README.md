# Post-Training and Alignment

Fine-tune and align small language models using the standard post-training pipeline.

## Current: SFT + Evals

- Fine-tune [Qwen 2.5 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) on [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) using LoRA
- Evaluate on HellaSwag and MMLU using lm-evaluation-harness
- Compare base vs fine-tuned model performance

## Setup

```bash
uv sync
```

## Usage

Configure `src/config.yaml`, then:

```bash
uv run src/main.py
```

Set `DEV: true` for quick pipeline tests, `DEV: false` for full runs.

## Tech Stack

Python 3.13+, PyTorch, Hugging Face (transformers, peft, trl, datasets), lm-evaluation-harness, Azure ML
