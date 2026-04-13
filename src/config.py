import os
from typing import Optional

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 0.3
MAX_SEQ_LENGTH = 512
OPTIMIZER = "adamw_torch"
FP16 = False
LOGGING_STEPS = 10
SEED = 42

OUTPUT_DIR = "outputs/model"
DATA_DIR = "outputs/data"
EVAL_DIR = "outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "Below is an instruction that describes a financial task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
)


def format_prompt(instruction: str, input_text: str = "", include_response: Optional[str] = None) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"
    if include_response is not None:
        prompt += include_response
    return prompt
