import torch
from src.config import format_prompt
from src.model import load_tokenizer, load_base_model, load_finetuned_model


def generate_answer(model, tokenizer, instruction: str, input_text: str = "",
                    max_new_tokens: int = 256, temperature: float = 0.1) -> str:
    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def compare_models(base_model, ft_model, tokenizer, question: str,
                   context: str = "", max_new_tokens: int = 256):
    print(f"QUESTION: {question}")

    base_ans = generate_answer(base_model, tokenizer, question, context, max_new_tokens)
    ft_ans = generate_answer(ft_model, tokenizer, question, context, max_new_tokens)

    print(f"\nBASE MODEL:\n{base_ans}")
    print(f"\nFINE-TUNED:\n{ft_ans}\n")

    return base_ans, ft_ans


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    base_model = load_base_model()
    base_model.eval()
    ft_model = load_finetuned_model(tokenizer)

    questions = [
        "What is a Roth IRA and how does it differ from a Traditional IRA?",
        "Explain the concept of compound interest with an example.",
        "What is the P/E ratio and how should investors interpret it?",
    ]

    for question in questions:
        compare_models(base_model, ft_model, tokenizer, question, max_new_tokens=300)
