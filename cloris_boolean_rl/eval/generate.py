"""Sampling completions from a model, on CPU or on a T4, through one path.

Imported lazily by everything that uses it. The deterministic half of the
eval must keep running on a machine with no torch installed, which is the
whole reason the baselines Cloris already has can be measured without a GPU.
"""

from __future__ import annotations

from typing import Sequence

from ..data.prompts import Prompt

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 200


def load_model(model_id: str, *, device: str | None = None, dtype: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = {}
    if dtype != "auto":
        kwargs["dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()
    return model, tokenizer


def make_generator(model, tokenizer, *, batch_size: int = 4, seed: int = 20260921):
    """A ``(prompts, samples, temperature) -> list[list[str]]`` callable."""
    import torch

    def generate(
        prompts: Sequence[Prompt], samples: int, temperature: float
    ) -> list[list[str]]:
        out: list[list[str]] = []
        device = next(model.parameters()).device
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            texts = [
                tokenizer.apply_chat_template(
                    prompt.messages, tokenize=False, add_generation_prompt=True
                )
                for prompt in chunk
            ]
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            torch.manual_seed(seed + start)
            with torch.no_grad():
                produced = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    num_return_sequences=samples,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = encoded["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                produced[:, width:], skip_special_tokens=True
            )
            for index in range(len(chunk)):
                out.append(decoded[index * samples : (index + 1) * samples])
        return out

    return generate
