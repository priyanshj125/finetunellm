# Financial QA Fine-Tuning

Fine-tuning Qwen2.5-1.5B-Instruct for financial question answering using LoRA in fp16. No BitsAndBytes dependency — runs reliably on a free Colab T4 GPU.

## Model

- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Method: LoRA (fp16, no quantization)
- LoRA rank: r=16, alpha=32
- Trainable parameters: approximately 1% of total

## Dataset

- Financial PhraseBank (sentiment analysis)
- FinGPT sentiment train split
- 20 hand-curated financial Q&A pairs covering core concepts
- Total: up to 2000 samples after cleaning and deduplication
- Split: 90% train, 10% test

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

## Inference

```bash
python -m src.inference
```

## Evaluation

Run from a notebook or script after training:

```python
from evaluation.evaluate import compute_scores, run_failure_cases
from evaluation.visualize import plot_loss_curve, plot_evaluation_metrics
```

## Repository Structure

```
repo/
  src/
    config.py          hyperparameters and prompt template
    dataset.py         data loading, cleaning, and splitting
    model.py           model and tokenizer loading, LoRA setup
    train.py           training pipeline
    inference.py       generation and comparison helpers
  data/
    curated_qa.py      20 hand-crafted financial Q&A samples
    data_loader.py     load preprocessed data from disk
  evaluation/
    evaluate.py        ROUGE, BLEU, and failure case analysis
    visualize.py       loss curve and metric comparison plots
  outputs/             generated at runtime (model, data, eval)
  requirements.txt
  README.md
  ARCHITECTURE.md
  DEMO_VIDEO.md
```

## Results

After 3 epochs on a T4 GPU (approximately 30 to 60 minutes):

| Metric  | Improvement |
|---------|-------------|
| ROUGE-1 | positive    |
| ROUGE-2 | positive    |
| ROUGE-L | positive    |
| BLEU    | positive    |

Exact numbers are written to `outputs/evaluation/results.json` after running evaluation.

## Known Limitations

- No access to real-time market data (prices, rates)
- Cannot provide legally binding personalized financial advice
- Occasional errors on precise arithmetic
- Subject to knowledge cutoff of the base Qwen model
# finetunellm
