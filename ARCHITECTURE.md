# Architecture

## Overview

This project fine-tunes a 1.5B parameter causal language model for financial question answering using parameter-efficient fine-tuning via LoRA. The base model weights remain frozen; only small rank-decomposition adapter matrices are trained.

## Base Model

Qwen2.5-1.5B-Instruct is a decoder-only transformer with strong instruction-following capability at a size that fits a free-tier GPU. It is loaded in fp16 precision without quantization.

## LoRA

Low-Rank Adaptation (LoRA) injects trainable matrices into the attention and MLP projection layers. For a weight matrix W, the update is represented as W + (B x A), where A and B are low-rank matrices with rank r=16. This reduces trainable parameters from approximately 1.5 billion to roughly 14 million (about 1%), while preserving full model expressiveness for the target domain.

Target modules:
- q_proj, k_proj, v_proj, o_proj (attention projections)
- gate_proj, up_proj, down_proj (MLP projections)

## Training

```
Input samples (instruction, input, output)
        |
        v
Alpaca-style prompt formatting
        |
        v
Tokenization (max 512 tokens, right-padded)
        |
        v
SFTTrainer with LoRA model
  - Optimizer: AdamW
  - Scheduler: cosine with 3% warmup
  - Effective batch size: 8 (2 per device x 4 gradient accumulation steps)
  - 3 epochs
        |
        v
LoRA adapter saved to outputs/model/
```

## Inference

The saved LoRA adapter is merged onto a freshly loaded base model at inference time using PeftModel.from_pretrained. Generation uses greedy decoding with a low temperature (0.1) and a repetition penalty of 1.1 to reduce loops.

## Evaluation

ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores are computed on up to 50 held-out test samples by comparing model-generated answers against reference outputs from the dataset. A separate failure case analysis tests known weaknesses: real-time data lookups, personalized advice, and precise arithmetic.

## Component Diagram

```
data/curated_qa.py
       +
HuggingFace datasets (Financial PhraseBank, FinGPT)
       |
       v
src/dataset.py  -->  outputs/data/train_dataset.json
                                  test_dataset.json
       |
       v
src/model.py  -->  Qwen2.5-1.5B-Instruct (fp16) + LoRA adapters
       |
       v
src/train.py  -->  SFTTrainer  -->  outputs/model/adapter_model.safetensors
       |
       v
src/inference.py  -->  generate_answer(), compare_models()
       |
       v
evaluation/evaluate.py   -->  outputs/evaluation/results.json
evaluation/visualize.py  -->  outputs/evaluation/training_loss_curve.png
                              outputs/evaluation/evaluation_metrics.png
```

## Hyperparameter Choices

| Parameter           | Value         | Reason                                              |
|---------------------|---------------|-----------------------------------------------------|
| Model               | Qwen2.5-1.5B  | Best instruction-following at a size that fits T4   |
| LoRA rank (r)       | 16            | Balance between expressiveness and overfitting      |
| LoRA alpha          | 32            | Standard 2x rank scaling                           |
| Learning rate       | 2e-4          | Standard for LoRA fine-tuning                      |
| Batch size (eff.)   | 8             | Fits T4 VRAM with gradient accumulation            |
| Max sequence length | 512           | Covers most financial Q&A pairs                    |
| Epochs              | 3             | Sufficient for 1500-sample dataset without overfit |
| Precision           | fp16          | Reduces VRAM; more stable than 4-bit on free Colab |
