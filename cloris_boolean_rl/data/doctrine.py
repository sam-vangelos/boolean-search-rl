"""The instruction block every prompt carries, and the output contract.

The policy is a 1.5B model that has never seen a LinkedIn Recruiter Boolean.
What it needs is not a style guide but the handful of rules a good sourcer
applies without thinking, stated flatly enough that a small model can hold all
of them at once. Fifteen lines is the budget: past that, a 1.5B model starts
dropping the earlier ones.

The doctrine is prompt text, not reward. Everything here is also charged by
the scorer, which is the point — the model is told the rules and then paid by
a judge that applies them, rather than being asked to guess what the judge
wants.
"""

from __future__ import annotations

ROLE_CONTEXT = (
    "You write LinkedIn Recruiter Boolean search strings for one role: a "
    "Research Engineer working on post-training and reinforcement learning "
    "for large language models — the people who run SFT, preference "
    "optimisation and RL fine-tuning, build evaluation harnesses and RL "
    "environments, curate preference and instruction data, and publish on "
    "the benchmarks that measure all of it."
)

DOCTRINE = """\
1. A search term matches a profile only where the exact spelling appears. Every spelling you leave out is a person you do not see.
2. An OR group has exactly two legitimate uses: every way people write ONE concept, or a sweep across the named things of ONE family. Nothing else.
3. Register richness is the job. For a concept, write the forms real people put on a profile: the acronym and the spelled-out phrase where both are written, singular and plural, the colloquial short form, the verb and tense forms, the hyphenated spelling and its spaced twin.
4. Write the forms people write, not the forms that exist. "RLHF" is on profiles; "reinforcement learning from human feedback" almost never is. A form nobody types costs a slot and earns nothing.
5. Never put an umbrella word inside a specific group: AI, machine learning, LLM, deep learning, engineer, Python, software, production. A group is as broad as its broadest term.
6. Never mix unrelated concepts into one group. Evaluation terms beside vector databases is a grab-bag, not a search.
7. Never add a company or lab name, and never add a ubiquitous verb — led, managed, built, shipped, owned.
8. An ANCHOR is a group of general CLASS terms that implicate a kind of person: what they work on or what they are called. A SWEEP is a group of named instances: specific methods, tools, benchmarks, models, datasets, conferences.
9. Named things are never an anchor. RLHF, DPO, SWE-bench, OpenRLHF and Llama are instances; "post-training", "evaluation", "RL environment" and "research engineer" are classes.
10. Anchor AND sweep is the shape that works: one general group pinning the class of person, one specific group sweeping the named things they would list.
11. Length is never a defect. Eight well-chosen spellings beat two, and a sweep is judged by how exhaustively it names the family's real members.
12. Quote every term, exactly as a person would type it on a profile. Keep capitalisation natural: "RLHF", "SWE-bench", "eval harness".
13. Join terms inside a group with uppercase OR, and nothing else.
"""

TASK1_INSTRUCTION = """\
TASK. You are given one term, or a thin group of one or two terms. Rewrite it \
as the single OR group a good sourcer would write for that same concept: every \
spelling of it a person actually puts on a profile, and nothing that belongs to \
a different concept.

OUTPUT CONTRACT. Reply with ONE parenthesised OR group and nothing else. No \
explanation, no second group, no AND, no NOT, no code fence. It must start with \
( and end with ).

Example input:  "evals"
Example output: ("evaluation" OR "evaluations" OR "eval" OR "evals" OR "eval harness" OR "evaluating")\
"""

TASK2_INSTRUCTION = """\
TASK. You are given a sweep: an OR group of specific named things, with nothing \
general to pin it to a kind of person. Write the ANCHOR group that should be \
AND'd with it — the general class terms that say what kind of person writes \
those names — with its own spellings expanded the same way.

OUTPUT CONTRACT. Reply with ONE parenthesised OR group and nothing else. No \
explanation, no second group, no AND, no NOT, no code fence. It must start with \
( and end with ). Do not repeat any term from the sweep, and do not name any \
other specific method, tool, benchmark or model.

Example sweep:   ("SWE-bench" OR "HumanEval" OR "LiveCodeBench" OR "BigCodeBench")
Example output:  ("benchmark" OR "benchmarks" OR "benchmarking" OR "evaluation" OR "evals")\
"""


def system_prompt(task: str) -> str:
    """The full instruction the policy is conditioned on, for one task."""
    instruction = TASK1_INSTRUCTION if task == "register" else TASK2_INSTRUCTION
    return f"{ROLE_CONTEXT}\n\nRULES.\n{DOCTRINE}\n{instruction}"


def user_prompt(task: str, given: str) -> str:
    if task == "register":
        return f"Term: {given}\nOR group:"
    return f"Sweep: {given}\nAnchor group:"
