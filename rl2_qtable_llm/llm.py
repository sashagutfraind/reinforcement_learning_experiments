"""
Local LLM helper.

`make_llm()` loads a model once via the transformers text-generation pipeline
and returns a plain callable:

    llm(messages, max_new_tokens=60, temperature=1.0) -> str

`messages` is the standard list-of-dicts chat format:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
"""

import os
import torch
import transformers
from transformers import pipeline

alt_HF_HOME = os.path.expanduser("~/icloud/huggingface")
if not os.path.exists(alt_HF_HOME):
    print(f"Using alternative HF_HOME location: {alt_HF_HOME}")
    os.environ["HF_HOME"] = alt_HF_HOME


MODEL = "Qwen/Qwen2.5-0.5B-Instruct" #https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

# Suppress noisy deprecation warnings from transformers internals
transformers.logging.set_verbosity_error()


def make_llm(model_name: str = MODEL):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Ensure pad_token_id is set — without it generation can hang waiting for padding
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    def call(messages: list[dict], max_new_tokens: int = 60, temperature: float = 1.0) -> str:
        do_sample = temperature > 0
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=1.0 if do_sample else None,
            top_k=20 if do_sample else None,
            min_p=0.0 if do_sample else None,
            repetition_penalty=1.0,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        # When messages are passed, generated_text is the full conversation list;
        # the assistant turn is the last entry.
        return outputs[0]["generated_text"][-1]["content"].strip()

    return call


def main():
    print("Loading model...")
    llm = make_llm()
    print("Model loaded.\n")

    tests = [
        {
            "label": "score extraction (temperature=0)",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rate how happy a boss sounds about a meeting being booked. "
                        "Reply with a single integer from 0 (very unhappy) to 10 (very happy). "
                        "No explanation, just the number."
                    ),
                },
                {"role": "user", "content": 'Boss said: "Monday morning? Are you kidding me?!"'},
            ],
            "max_new_tokens": 5,
            "temperature": 0.0,
        },
        {
            "label": "score extraction — positive reaction",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rate how happy a boss sounds about a meeting being booked. "
                        "Reply with a single integer from 0 (very unhappy) to 10 (very happy). "
                        "No explanation, just the number."
                    ),
                },
                {"role": "user", "content": 'Boss said: "Friday at 3pm works perfectly, thank you!"'},
            ],
            "max_new_tokens": 5,
            "temperature": 0.0,
        },
        {
            "label": "short chat (temperature=1.0)",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                {"role": "user", "content": "What day of the week is best for a meeting?"},
            ],
            "max_new_tokens": 60,
            "temperature": 1.0,
        },
    ]

    for t in tests:
        print(f"--- {t['label']} ---")
        result = llm(t["messages"], max_new_tokens=t["max_new_tokens"], temperature=t["temperature"])
        print(f"Response: {result!r}\n")


if __name__ == "__main__":
    main()
