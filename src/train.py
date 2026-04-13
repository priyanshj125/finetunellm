import json
from transformers import TrainerCallback, set_seed
from trl import SFTTrainer, SFTConfig
from src.config import (
    OUTPUT_DIR, NUM_EPOCHS, PER_DEVICE_BATCH_SIZE, GRAD_ACCUM_STEPS,
    LEARNING_RATE, LR_SCHEDULER, WARMUP_RATIO, MAX_GRAD_NORM,
    OPTIMIZER, FP16, LOGGING_STEPS, MAX_SEQ_LENGTH, SEED,
    format_prompt,
)
from src.dataset import build_dataset
from src.model import load_tokenizer, load_base_model, apply_lora


class LossLogger(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)
        if "eval_loss" in logs:
            self.eval_losses.append({"step": state.global_step, "loss": logs["eval_loss"]})


def formatting_func(example: dict) -> list:
    instructions = example["instruction"]
    inputs = example.get("input", []) or []
    outputs = example["output"]

    if not isinstance(instructions, list):
        instructions = [instructions]
        inputs = [inputs]
        outputs = [outputs]

    if len(inputs) < len(instructions):
        inputs = inputs + [""] * (len(instructions) - len(inputs))

    results = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        inst = str(inst) if inst else ""
        inp = str(inp) if inp else ""
        out = str(out) if out else ""
        results.append(format_prompt(instruction=inst, input_text=inp, include_response=out))
    return results


def main():
    set_seed(SEED)

    tokenizer = load_tokenizer()
    train_hf, test_hf, train_samples, test_samples = build_dataset()
    model = load_base_model()
    model, peft_config = apply_lora(model)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=False,
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=0.001,
        fp16=FP16,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=LOGGING_STEPS,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=SEED,
    )

    loss_logger = LossLogger()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_hf,
        eval_dataset=test_hf,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_func,
        callbacks=[loss_logger],
    )

    train_result = trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "precision": "fp16",
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "num_train_samples": len(train_hf),
        "num_eval_samples": len(test_hf),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_r": 16,
        "lora_alpha": 32,
    }
    with open(f"{OUTPUT_DIR}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Loss: {train_result.training_loss:.4f}")
    return trainer, loss_logger, train_result, tokenizer, test_samples


if __name__ == "__main__":
    main()
