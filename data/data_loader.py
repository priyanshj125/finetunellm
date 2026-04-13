import json
from datasets import Dataset
from src.config import DATA_DIR


def load_from_disk():
    with open(f"{DATA_DIR}/train_dataset.json") as f:
        train_samples = json.load(f)
    with open(f"{DATA_DIR}/test_dataset.json") as f:
        test_samples = json.load(f)

    train_hf = Dataset.from_list(train_samples)
    test_hf = Dataset.from_list(test_samples)

    print(f"Loaded train: {len(train_samples)} | test: {len(test_samples)}")
    return train_hf, test_hf, train_samples, test_samples
