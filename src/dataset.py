import re
import json
import random
from datasets import load_dataset, Dataset
from src.config import SEED, DATA_DIR
from data.curated_qa import CURATED_QA


def load_raw_samples():
    raw_samples = []

    try:
        fpb = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        for item in fpb["train"]:
            sentiment = label_map[item["label"]]
            raw_samples.append({
                "instruction": "Analyze the sentiment of the following financial news sentence and explain your reasoning.",
                "input": item["sentence"],
                "output": (
                    f"The sentiment of this financial statement is {sentiment}. "
                    f"The sentence conveys a {sentiment} outlook for the company or market."
                ),
            })
    except Exception as e:
        print(f"Financial PhraseBank unavailable: {e}")

    try:
        fingpt = load_dataset("FinGPT/fingpt-sentiment-train", trust_remote_code=True)
        for item in list(fingpt["train"])[:800]:
            if item.get("input") and item.get("output"):
                raw_samples.append({
                    "instruction": item.get("instruction", "Analyze the following financial text."),
                    "input": item["input"],
                    "output": item["output"],
                })
    except Exception as e:
        print(f"FinGPT unavailable: {e}")

    return raw_samples


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def is_valid_sample(sample: dict) -> bool:
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    if not instruction or not output:
        return False
    if len(instruction) < 10 or len(output) < 20:
        return False
    if len(instruction) + len(output) > 3000:
        return False
    if output.strip() == instruction.strip():
        return False
    return True


def build_dataset(max_samples: int = 2000):
    raw_samples = load_raw_samples()
    all_samples = CURATED_QA + raw_samples

    cleaned = []
    for s in all_samples:
        c = {
            "instruction": clean_text(s.get("instruction", "")),
            "input": clean_text(s.get("input", "")),
            "output": clean_text(s.get("output", "")),
        }
        if is_valid_sample(c):
            cleaned.append(c)

    seen = set()
    deduped = []
    for s in cleaned:
        key = s["instruction"].lower().strip()[:100]
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    random.seed(SEED)
    random.shuffle(deduped)
    final = deduped[:max_samples]

    split_idx = int(len(final) * 0.9)
    train_samples = final[:split_idx]
    test_samples = final[split_idx:]

    with open(f"{DATA_DIR}/train_dataset.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(f"{DATA_DIR}/test_dataset.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    train_hf = Dataset.from_list(train_samples)
    test_hf = Dataset.from_list(test_samples)

    print(f"Train: {len(train_samples)} | Test: {len(test_samples)}")
    return train_hf, test_hf, train_samples, test_samples
