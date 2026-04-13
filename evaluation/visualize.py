import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import EVAL_DIR


def plot_loss_curve(loss_logger):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Metrics - Financial QA Fine-Tuning\nQwen2.5-1.5B-Instruct + LoRA (fp16)",
                 fontsize=14, fontweight="bold")

    ax1 = axes[0]
    if loss_logger.train_losses:
        ax1.plot(loss_logger.steps, loss_logger.train_losses,
                 color="#2196F3", linewidth=2, label="Train Loss", alpha=0.8)
        if len(loss_logger.train_losses) > 10:
            window = min(10, len(loss_logger.train_losses) // 3)
            smoothed = pd.Series(loss_logger.train_losses).rolling(window=window).mean()
            ax1.plot(loss_logger.steps, smoothed,
                     color="#F44336", linewidth=2.5, label=f"Smoothed (w={window})", linestyle="--")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    ax2 = axes[1]
    if loss_logger.eval_losses:
        eval_steps = [e["step"] for e in loss_logger.eval_losses]
        eval_vals = [e["loss"] for e in loss_logger.eval_losses]
        ax2.plot(eval_steps, eval_vals,
                 color="#4CAF50", linewidth=2.5, marker="o", markersize=8, label="Eval Loss")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Eval Loss")
        ax2.set_title("Validation Loss per Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#f8f9fa")

    plt.tight_layout()
    plot_path = f"{EVAL_DIR}/training_loss_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Loss curve saved: {plot_path}")


def plot_evaluation_metrics(base_avg, ft_avg):
    metrics_list = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
    keys = ["rouge1", "rouge2", "rougeL", "bleu"]
    base_vals = [base_avg[k] for k in keys]
    ft_vals = [ft_avg[k] for k in keys]
    x = np.arange(len(metrics_list))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Financial QA - Base vs Fine-Tuned Model Evaluation\nQwen2.5-1.5B-Instruct + LoRA (fp16)",
                 fontsize=13, fontweight="bold")

    bars1 = ax1.bar(x - width / 2, base_vals, width, label="Base Model", color="#2196F3", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, ft_vals, width, label="Fine-Tuned", color="#4CAF50", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.002,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_list)
    ax1.set_ylabel("Score")
    ax1.set_title("Metric Comparison")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_facecolor("#f8f9fa")
    ax1.set_ylim(0, max(max(base_vals), max(ft_vals)) * 1.2)

    improvements = [(ft - b) / b * 100 if b > 0 else 0 for b, ft in zip(base_vals, ft_vals)]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in improvements]
    bars3 = ax2.bar(metrics_list, improvements, color=colors, alpha=0.85)

    for bar, val in zip(bars3, improvements):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + (0.5 if h >= 0 else -1.5),
                 f"{val:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_ylabel("Relative Improvement (%)")
    ax2.set_title("Improvement Over Base Model")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    plt.tight_layout()
    eval_plot_path = f"{EVAL_DIR}/evaluation_metrics.png"
    plt.savefig(eval_plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Evaluation chart saved: {eval_plot_path}")
