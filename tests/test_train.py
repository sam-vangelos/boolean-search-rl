"""The training rig's wiring, checked without loading a model.

Everything here is the part of a GRPO run that can be wrong before a GPU is
rented: the reward callable's signature and ordering, the guards on the two
settings that silently ruin a run, and whether the notebook still says what
the module says.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cloris_boolean_rl.data.prompts import TRAIN, load_prompts, select
from cloris_boolean_rl.reward import tasks
from cloris_boolean_rl.train import colab_first_run, grpo

ROOT = Path(__file__).resolve().parents[1]


def test_beta_may_not_be_zero():
    with pytest.raises(SystemExit):
        grpo.parse_args(["--beta", "0"])


def test_temperature_may_not_collapse_the_group():
    with pytest.raises(SystemExit):
        grpo.parse_args(["--temperature", "0.2"])


def test_the_default_run_is_the_one_the_readme_describes():
    config = grpo.parse_args([])
    assert config.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.generations == 8
    assert config.lora_r == 16
    assert config.prompts == 300
    assert config.beta > 0
    assert config.temperature >= 0.8
    assert config.max_completion_length <= 200


def test_the_smoke_is_small_enough_to_be_a_smoke():
    config = grpo.parse_args(["--smoke"])
    assert config.model == grpo.SMOKE_MODEL
    assert config.max_steps == 2
    assert not config.four_bit and not config.unsloth
    assert config.prompts <= 8


def test_the_generation_batch_divides_by_the_group_size():
    """TRL refuses a generation batch that is not a multiple of G, and it
    refuses it after the model is on the GPU."""
    for config in (grpo.parse_args([]), grpo.parse_args(["--smoke"])):
        batch = config.per_device_batch * config.grad_accum
        assert batch % config.generations == 0, config.output_dir


def test_training_prompts_come_only_from_the_train_split():
    config = grpo.parse_args([])
    prompts = grpo.training_prompts(config)
    assert prompts
    assert {p.split for p in prompts} == {TRAIN}
    assert len(prompts) <= config.prompts


def test_the_reward_callable_matches_completions_to_their_prompts():
    prompts = select(load_prompts(), split=TRAIN, limit=6)
    reward_fn = grpo.make_reward_function(prompts)
    completions = [p.reference for p in prompts]
    ids = [p.id for p in prompts]

    # Shuffled, because TRL does not promise dataset order.
    order = [3, 0, 5, 1, 4, 2]
    values = reward_fn(
        completions=[completions[i] for i in order],
        prompt_id=[ids[i] for i in order],
    )
    assert values == [tasks.reward(prompts[i], completions[i]) for i in order]


def test_the_reward_callable_reads_conversational_completions():
    prompts = select(load_prompts(), split=TRAIN, limit=1)
    reward_fn = grpo.make_reward_function(prompts)
    text = prompts[0].reference
    as_messages = [[{"role": "assistant", "content": text}]]
    assert reward_fn(completions=as_messages, prompt_id=[prompts[0].id]) == reward_fn(
        completions=[text], prompt_id=[prompts[0].id]
    )


def test_the_reward_callable_survives_garbage():
    prompts = select(load_prompts(), split=TRAIN, limit=1)
    reward_fn = grpo.make_reward_function(prompts)
    floor = tasks.reward_floor_for(prompts[0])
    for junk in ("", "I'm sorry", "((((", [], [{"role": "assistant"}]):
        values = reward_fn(completions=[junk], prompt_id=[prompts[0].id])
        assert values == [floor]


def test_the_dataset_carries_the_prompt_id():
    pytest.importorskip("datasets")
    prompts = select(load_prompts(), split=TRAIN, limit=4)
    dataset = grpo.build_dataset(prompts)
    assert dataset.column_names == ["prompt", "prompt_id"]
    assert dataset[0]["prompt_id"] == prompts[0].id
    assert dataset[0]["prompt"][0]["role"] == "system"


def test_the_notebook_is_generated_from_the_module():
    path = ROOT / "notebooks" / "colab_grpo_first_run.ipynb"
    assert path.exists(), "run colab_first_run.py --write-notebook"
    saved = json.loads(path.read_text())
    code = [cell for cell in saved["cells"] if cell["cell_type"] == "code"]
    assert len(code) == len(colab_first_run.CELLS)
    for cell, (name, source) in zip(code, colab_first_run.CELLS):
        assert cell["metadata"]["name"] == name
        assert "".join(cell["source"]) == source


def test_the_notebook_runs_the_tests_before_it_trains():
    order = [name for name, _ in colab_first_run.CELLS]
    assert order.index("verify") < order.index("train")
    assert order.index("smoke") < order.index("train")
    assert order.index("baselines") < order.index("eval")


def test_the_unsloth_path_leaves_fp16_to_unsloth():
    """On a T4, fp16=True under Unsloth crashed step 0: the fp16 GradScaler was
    handed bf16 gradients by Unsloth's patched loop. Plain transformers keeps
    fp16, which is what trains there."""
    unsloth = grpo.grpo_args(grpo.RunConfig(unsloth=True), cuda=True)
    plain = grpo.grpo_args(grpo.RunConfig(unsloth=False), cuda=True)
    assert unsloth["fp16"] is False and unsloth["bf16"] is False
    assert plain["fp16"] is True and plain["bf16"] is False


def test_run_two_knobs_reach_the_trainer():
    """Run 1 barely moved the policy (lr 1e-5, linear decay to zero, KL under
    0.03). Run 2's schedule, batch and completion cap are flags, and each one
    has to arrive in the GRPOConfig kwargs, not just the RunConfig."""
    config = grpo.parse_args(
        ["--grad-accum", "4", "--warmup-steps", "10", "--lr-scheduler", "cosine",
         "--max-completion-length", "128", "--learning-rate", "5e-5"]
    )
    assert config.grad_accum == 4
    assert config.warmup_steps == 10
    assert config.lr_scheduler == "cosine"
    assert config.max_completion_length == 128
    assert config.learning_rate == 5e-5
    args = grpo.grpo_args(config, cuda=True)
    assert args["gradient_accumulation_steps"] == 4
    assert args["warmup_steps"] == 10
    assert args["lr_scheduler_type"] == "cosine"
    assert args["learning_rate"] == 5e-5
    assert args["max_completion_length"] == 128


def test_batch_and_lora_flags_reach_the_config():
    config = grpo.parse_args(
        ["--per-device-batch", "16", "--lora-r", "32", "--lora-alpha", "64"]
    )
    assert config.per_device_batch == 16
    assert config.lora_r == 32
    assert config.lora_alpha == 64
    assert grpo.grpo_args(config, cuda=True)["per_device_train_batch_size"] == 16


def test_run_one_defaults_are_unchanged():
    """Run 1 has to stay reproducible from its run_config.json."""
    args = grpo.grpo_args(grpo.RunConfig(), cuda=True)
    assert args["lr_scheduler_type"] == "linear"
    assert args["warmup_steps"] == 0
    assert args["learning_rate"] == 1e-5
    assert args["gradient_accumulation_steps"] == 2
    assert args["per_device_train_batch_size"] == 8
    assert args["max_completion_length"] == 160


# ---------------------------------------------------------------------------
# Resuming from a checkpoint
# ---------------------------------------------------------------------------


def test_resume_from_is_off_unless_asked_for():
    assert grpo.parse_args([]).resume_from is None
    assert grpo.parse_args(["--smoke"]).resume_from is None
    assert grpo.parse_args(["--smoke"]) == grpo.SMOKE


def test_resume_from_reaches_the_config():
    config = grpo.parse_args(
        ["--resume-from", "runs/re-v1/checkpoint-75", "--output-dir", "runs/re-v1",
         "--max-steps", "100"]
    )
    assert config.resume_from == "runs/re-v1/checkpoint-75"
    assert config.output_dir == "runs/re-v1"
    assert config.max_steps == 100


def _fake_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "trainer_state.json").write_text("{}")
    return path


def test_a_relative_checkpoint_resolves_against_the_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr(grpo, "ROOT", tmp_path)
    checkpoint = _fake_checkpoint(tmp_path / "runs" / "re-v1" / "checkpoint-75")
    config = grpo.RunConfig(resume_from="runs/re-v1/checkpoint-75")
    assert grpo.resolve_checkpoint(config) == checkpoint
    assert grpo.resolve_checkpoint(grpo.RunConfig()) is None


def test_an_absolute_checkpoint_is_taken_as_given(tmp_path):
    checkpoint = _fake_checkpoint(tmp_path / "checkpoint-25")
    config = grpo.RunConfig(resume_from=str(checkpoint))
    assert grpo.resolve_checkpoint(config) == checkpoint


def test_a_missing_checkpoint_fails_before_any_model_loads(monkeypatch, tmp_path):
    def must_not_build(*_args, **_kwargs):
        raise AssertionError("built a trainer for a checkpoint that is not there")

    monkeypatch.setattr(grpo, "build_trainer", must_not_build)
    output = tmp_path / "out"
    config = grpo.RunConfig(
        output_dir=str(output), resume_from=str(tmp_path / "checkpoint-75")
    )
    with pytest.raises(FileNotFoundError, match="checkpoint-75"):
        grpo.run(config)
    assert not output.exists(), "a failed resume must not write a run directory"


def test_a_run_directory_is_not_a_checkpoint(tmp_path):
    """The easy mistake is ``--resume-from runs/re-v1``: the run directory
    exists, but it holds checkpoints rather than being one."""
    run_dir = tmp_path / "re-v1"
    _fake_checkpoint(run_dir / "checkpoint-75")
    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        grpo.resolve_checkpoint(grpo.RunConfig(resume_from=str(run_dir)))


@pytest.mark.skipif(
    not os.environ.get("CLORIS_RL_SLOW"),
    reason="loads Qwen2.5-0.5B and trains on CPU for ~3 min; set CLORIS_RL_SLOW=1",
)
def test_a_resumed_smoke_picks_up_where_the_checkpoint_left_off(tmp_path):
    """The real thing: two smoke steps, a checkpoint at step 2, then a second
    process-equivalent run that resumes from it to step 3. The resumed run
    must train one step, not three, and its history must start with the
    first run's two steps rather than a fresh step 1."""
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    output = tmp_path / "smoke"
    first = grpo.run(
        grpo.RunConfig(
            **{**grpo.SMOKE.to_dict(), "output_dir": str(output), "max_steps": 2,
               "save_steps": 2}
        )
    )
    checkpoint = output / "checkpoint-2"
    assert checkpoint.is_dir()
    assert first["steps"] == 2
    assert first["resumed_from"] is None
    assert [row["step"] for row in first["log_history"]] == [1, 2]

    second = grpo.run(
        grpo.RunConfig(
            **{**grpo.SMOKE.to_dict(), "output_dir": str(output), "max_steps": 3,
               "save_steps": 2, "resume_from": str(checkpoint)}
        )
    )
    assert second["steps"] == 3
    assert second["resumed_from"] == str(checkpoint)
    steps = [row["step"] for row in second["log_history"]]
    assert steps == [1, 2, 3], steps
    # Steps 1 and 2 came from the checkpoint's trainer state, not a retrain.
    assert second["log_history"][:2] == first["log_history"]
    saved = json.loads((output / "train_summary.json").read_text())
    assert saved["resumed_from"] == str(checkpoint)
