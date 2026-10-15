"""GRPO on the verifiable reward: the Colab T4 configuration, and a CPU smoke.

Why GRPO and not PPO: the reward is a verifier, not a learned critic, and
GRPO throws the critic away and normalises advantages inside the group of G
completions of ONE prompt. That is exactly the comparison the reward is good
at. The scorer's composite is a product of five terms, so a two-point gain on
a bad string is invisible in absolute terms and obvious against its siblings;
``log_composite`` plus within-group normalisation is the whole argument for
this algorithm on this problem.

Three settings are deliberate rather than inherited.

KL beta stays above zero. The temptation on a verifiable reward is to set it
to zero and let the policy run, and on a reward with a known padding hack
that is how you get a model that has forgotten how to write English and
speaks only in OR groups. The gates refuse the hack; beta keeps the policy
near a model that can still read the prompt.

Temperature stays at or above 0.8. GRPO learns from the spread inside the
group: at temperature 0.2 the eight completions are the same string, every
advantage is zero, and the run is an expensive no-op.

Diversity is logged and never rewarded. ``eval/metrics.py`` says why.

``--smoke`` runs two steps on CPU with the 0.5B model in full precision. It
proves the loop — dataset, chat template, generation, reward, backward pass,
checkpoint — end to end without a GPU, which is what makes the first Colab
cell a confirmation rather than a debugging session.

``--resume-from`` exists because free Colab VMs get reclaimed, one of them
four minutes short of step 100. Point it at a checkpoint the trainer wrote,
into the same run directory: ``--resume-from runs/re-v1/checkpoint-75
--output-dir runs/re-v1``. ``--max-steps`` is the absolute target, not the
remainder, so resuming that 100-step run with ``--max-steps 100`` trains 25
more steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloris_boolean_rl.data.prompts import (  # noqa: E402
    TRAIN,
    Prompt,
    load_prompts,
    select,
)
from cloris_boolean_rl.reward import tasks  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 20260921


@dataclass(frozen=True)
class RunConfig:
    """Everything a run is, in one object the report can print verbatim."""

    model: str = DEFAULT_MODEL
    output_dir: str = "runs/re-v1"
    prompts: int = 300
    generations: int = 8
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 1e-5
    # Run 1's schedule: no warmup, linear decay to zero. Kept as the default so
    # its run_config.json still reproduces it.
    warmup_steps: int = 0
    lr_scheduler: str = "linear"
    beta: float = 0.04
    temperature: float = 0.9
    top_p: float = 0.95
    max_completion_length: int = 160
    per_device_batch: int = 8
    grad_accum: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    save_steps: int = 25
    logging_steps: int = 1
    four_bit: bool = True
    unsloth: bool = True
    seed: int = SEED
    smoke: bool = False
    resume_from: str | None = None

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


SMOKE = RunConfig(
    model=SMOKE_MODEL,
    output_dir="runs/smoke",
    prompts=4,
    generations=4,
    max_steps=2,
    max_completion_length=48,
    per_device_batch=4,
    grad_accum=1,
    save_steps=2,
    four_bit=False,
    unsloth=False,
    smoke=True,
)


# ---------------------------------------------------------------------------
# Dataset and reward
# ---------------------------------------------------------------------------


def build_dataset(prompts: Sequence[Prompt]):
    from datasets import Dataset

    return Dataset.from_list(
        [{"prompt": prompt.messages, "prompt_id": prompt.id} for prompt in prompts]
    )


def _text_of(completion) -> str:
    """TRL hands back messages for a conversational dataset, strings for a
    plain one. The reward reads one string either way."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def make_reward_function(prompts: Sequence[Prompt]):
    """The TRL reward callable, closed over the prompt set by id.

    It is pure and takes no lock: the scorer is stdlib, allocates nothing
    global, and runs at roughly 40 microseconds a call, so G=8 over 300
    prompts costs well under a second per epoch and never becomes the thing
    the run is waiting on.
    """
    by_id = {prompt.id: prompt for prompt in prompts}

    def boolean_scorer_reward(completions, prompt_id, **_kwargs) -> list[float]:
        return [
            tasks.reward(by_id[pid], _text_of(completion))
            for completion, pid in zip(completions, prompt_id)
        ]

    boolean_scorer_reward.__name__ = "boolean_scorer_reward"
    return boolean_scorer_reward


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_policy(config: RunConfig):
    """Unsloth on a GPU, plain transformers everywhere else.

    Unsloth's 4-bit path needs CUDA and does not resolve on macOS arm64, so
    the smoke deliberately does not use it — and the loop it proves is the
    same loop, because everything Unsloth changes sits below the trainer.
    """
    import torch

    cuda = torch.cuda.is_available()
    if config.unsloth and cuda:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model,
            max_seq_length=1536,
            load_in_4bit=config.four_bit,
            fast_inference=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        return model, tokenizer, None

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype=torch.float16 if cuda else torch.float32
    )
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    return model, tokenizer, peft_config


def grpo_args(config: RunConfig, cuda: bool) -> dict:
    """The ``GRPOConfig`` keyword arguments, built without torch or a model.

    A plain dict rather than the config object, because ``TrainingArguments``
    refuses ``fp16=True`` on a machine with no GPU, and the precision choice
    is exactly the thing worth testing on one.
    """
    report_to = ["wandb"] if os.environ.get("CLORIS_RL_WANDB") else []
    return dict(
        output_dir=str(ROOT / config.output_dir),
        num_generations=config.generations,
        per_device_train_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler,
        # Never zero. A verifiable reward with a known padding hack will walk
        # away from the base model if nothing holds it.
        beta=config.beta,
        temperature=config.temperature,
        top_p=config.top_p,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        log_completions=True,
        num_completions_to_print=2,
        report_to=report_to,
        seed=config.seed,
        bf16=False,
        # fp16 under Unsloth crashes step 0 on a T4: the GradScaler is handed
        # bf16 gradients. Unsloth manages its own precision; leave it alone.
        fp16=cuda and not config.unsloth,
        gradient_checkpointing=cuda,
        use_vllm=False,
        # The smoke says CPU and means CPU. Left to itself the Trainer picks
        # MPS on an Apple machine, where generation for a model this small ran
        # at roughly 200 seconds a step against about 25 on the CPU — op
        # fallbacks and per-call synchronisation, not compute. A smoke that
        # takes ten minutes is a smoke nobody runs.
        use_cpu=config.smoke and not cuda,
        dataloader_pin_memory=cuda,
    )


def build_trainer(config: RunConfig, prompts: Sequence[Prompt]):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    args = GRPOConfig(**grpo_args(config, torch.cuda.is_available()))
    model, tokenizer, peft_config = load_policy(config)
    return GRPOTrainer(
        model=model,
        reward_funcs=make_reward_function(prompts),
        args=args,
        train_dataset=build_dataset(prompts),
        processing_class=tokenizer,
        peft_config=peft_config,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_checkpoint(config: RunConfig) -> Path | None:
    """The checkpoint to resume from, resolved against ROOT like
    ``output_dir``, or None for a fresh run.

    It fails here rather than inside ``trainer.train`` because by then the
    model is loaded, and a typo should not cost a model load to find. A
    directory without ``trainer_state.json`` is refused too: that is what
    passing the run directory (``runs/re-v1``) instead of one of its
    checkpoints looks like, and what a half-downloaded checkpoint looks like.
    """
    if config.resume_from is None:
        return None
    checkpoint = ROOT / config.resume_from
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"no checkpoint directory at {checkpoint}")
    if not (checkpoint / "trainer_state.json").is_file():
        raise FileNotFoundError(
            f"{checkpoint} has no trainer_state.json; point --resume-from at a "
            "checkpoint-N directory inside the run, not the run itself"
        )
    return checkpoint


def training_prompts(config: RunConfig) -> list[Prompt]:
    return select(load_prompts(), split=TRAIN, limit=config.prompts, seed=config.seed)


def run(config: RunConfig) -> dict:
    import time

    checkpoint = resolve_checkpoint(config)
    prompts = training_prompts(config)
    output = ROOT / config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps({**config.to_dict(), "prompt_ids": [p.id for p in prompts]}, indent=2)
        + "\n"
    )

    trainer = build_trainer(config, prompts)
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    elapsed = time.perf_counter() - started

    trainer.save_model(str(output / "adapter"))
    history = [row for row in trainer.state.log_history if "reward" in row]
    summary = {
        "config": config.to_dict(),
        "prompts": len(prompts),
        "steps": trainer.state.global_step,
        "seconds": round(elapsed, 2),
        "log_history": history,
        "adapter": str(output / "adapter"),
        "resumed_from": str(checkpoint) if checkpoint else None,
    }
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 CPU steps, 0.5B model")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompts", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument(
        "--lr-scheduler", default=None, help="a transformers scheduler name, e.g. cosine"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None, help="prompts per step = batch x this / G"
    )
    parser.add_argument("--per-device-batch", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--no-unsloth", action="store_true")
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="PATH",
        help="a checkpoint-N directory to resume from; --max-steps stays the total",
    )
    args = parser.parse_args(argv)

    base = SMOKE if args.smoke else RunConfig()
    overrides = {
        key: value
        for key, value in {
            "model": args.model,
            "output_dir": args.output_dir,
            "prompts": args.prompts,
            "generations": args.generations,
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "beta": args.beta,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler": args.lr_scheduler,
            "grad_accum": args.grad_accum,
            "per_device_batch": args.per_device_batch,
            "max_completion_length": args.max_completion_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "save_steps": args.save_steps,
            "resume_from": args.resume_from,
        }.items()
        if value is not None
    }
    if args.no_unsloth:
        overrides["unsloth"] = False
    config = RunConfig(**{**base.to_dict(), **overrides})
    if config.beta <= 0:
        raise SystemExit("beta must stay above zero; see this module's docstring")
    if config.temperature < 0.8:
        raise SystemExit("temperature below 0.8 collapses the GRPO group")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
