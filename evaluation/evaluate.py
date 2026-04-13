import json
import numpy as np
from rouge_score import rouge_scorer as rs_module
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.config import EVAL_DIR
from src.inference import generate_answer


def compute_scores(base_model, ft_model, tokenizer, test_samples, eval_limit=50):
    eval_subset = test_samples[:eval_limit]
    scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method4

    base_scores = {"rouge1": [], "rouge2": [], "rougeL": [], "bleu": []}
    ft_scores = {"rouge1": [], "rouge2": [], "rougeL": [], "bleu": []}

    for i, sample in enumerate(eval_subset):
        if i % 10 == 0:
            print(f"Evaluating {i+1}/{len(eval_subset)}")

        reference = sample["output"]
        base_pred = generate_answer(base_model, tokenizer, sample["instruction"],
                                    sample.get("input", ""), max_new_tokens=200)
        ft_pred = generate_answer(ft_model, tokenizer, sample["instruction"],
                                  sample.get("input", ""), max_new_tokens=200)

        for model_scores, pred in [(base_scores, base_pred), (ft_scores, ft_pred)]:
            rouge_result = scorer.score(reference, pred)
            model_scores["rouge1"].append(rouge_result["rouge1"].fmeasure)
            model_scores["rouge2"].append(rouge_result["rouge2"].fmeasure)
            model_scores["rougeL"].append(rouge_result["rougeL"].fmeasure)
            ref_tokens = reference.lower().split()
            pred_tokens = pred.lower().split()
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth) if pred_tokens else 0.0
            model_scores["bleu"].append(bleu)

    base_avg = {k: np.mean(v) for k, v in base_scores.items()}
    ft_avg = {k: np.mean(v) for k, v in ft_scores.items()}

    print(f"\n{'Metric':<12} {'Base':>10} {'Fine-Tuned':>12} {'Delta':>10}")
    for metric in ["rouge1", "rouge2", "rougeL", "bleu"]:
        b = base_avg[metric]
        f = ft_avg[metric]
        d = f - b
        pct = d / b * 100 if b > 0 else 0
        print(f"{metric.upper():<12} {b:>10.4f} {f:>12.4f} {d:>+8.4f} ({pct:+.1f}%)")

    results = {
        "eval_samples": len(eval_subset),
        "base_model": base_avg,
        "fine_tuned_model": ft_avg,
        "improvements": {
            m: {"absolute": ft_avg[m] - base_avg[m],
                "relative_pct": ((ft_avg[m] - base_avg[m]) / base_avg[m] * 100) if base_avg[m] > 0 else 0}
            for m in ["rouge1", "rouge2", "rougeL", "bleu"]
        },
    }

    results_path = f"{EVAL_DIR}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return base_avg, ft_avg


def run_failure_cases(ft_model, tokenizer):
    failure_questions = [
        {"question": "What was the exact stock price of Apple on March 15, 2023?",
         "expected_failure": "Real-time historical price data"},
        {"question": "Should I invest my entire savings in Tesla stock right now?",
         "expected_failure": "Personalized investment advice"},
        {"question": "Calculate the exact compound interest on $50,432.17 at 6.73% for 7.5 years with monthly compounding.",
         "expected_failure": "Precise numerical computation"},
        {"question": "What is the current federal funds rate?",
         "expected_failure": "Real-time data"},
    ]

    failure_results = []
    for fc in failure_questions:
        print(f"\nExpected failure: {fc['expected_failure']}")
        print(f"Q: {fc['question']}")
        response = generate_answer(ft_model, tokenizer, fc["question"], max_new_tokens=150)
        print(f"A: {response[:300]}")
        failure_results.append({
            "failure_type": fc["expected_failure"],
            "question": fc["question"],
            "model_response": response,
        })

    failure_path = f"{EVAL_DIR}/failure_cases.json"
    with open(failure_path, "w") as f:
        json.dump(failure_results, f, indent=2)
    print(f"\nFailure cases saved: {failure_path}")

    return failure_results
