"""LLM-as-judge for the LongMemEval-S QA arm.

A faithful re-implementation of the official evaluation
(``xiaowu0162/LongMemEval`` -> ``src/evaluation/evaluate_qa.py``,
function ``get_anscheck_prompt``). Comparability with published numbers
comes from the *protocol* -- the per-type correctness prompts reproduced
verbatim below, ``temperature=0``, the yes/no label rule, and the
abstention branch -- not from any specific judge model:

- ``temperature=0``, ``max_tokens=10``
- correctness label = whether the judge's reply contains "yes"
- per-question-type correctness criteria, plus a dedicated abstention
  branch selected whenever ``"_abs"`` appears in the ``question_id``

The original paper used ``gpt-4o-2024-08-06`` and measured >97% agreement
with human experts (Table 6). We deliberately do **not** pin that now-dated
model: the judge is a constrained yes/no grader, so a current frontier
model agrees with humans at least as well and only improves grading
fidelity. The default is a frontier remote model for independence from the
(local) reader; point ``--judge-model`` at any litellm-supported model
(including a local one) for cheap iteration. The one caveat to document in
the paper: numbers graded by a frontier judge are not byte-identical to the
gpt-4o-graded leaderboard, though the constrained protocol keeps them close.
"""

from __future__ import annotations

import logging

from litellm import completion

logger = logging.getLogger(__name__)

# Frontier remote judge by default, for independence from the (local) reader.
# Overridable via --judge-model (e.g. a local ollama model for key-free
# iteration). Routed through litellm; needs the matching provider API key.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for "
    "the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
    "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the "
    "points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from "
    "a model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the "
    "question as unanswerable? Answer yes or no only."
)


def is_abstention(question_id: str, question_type: str = "") -> bool:
    """Abstention iff ``"_abs"`` marks the id (matches the official script).

    ``question_type`` is also checked because some dataset releases carry
    the ``_abs`` suffix on the type rather than (or in addition to) the id.
    """
    return "_abs" in question_id or question_type.endswith("_abs")


def build_judge_prompt(
    *,
    question: str,
    answer: str,
    hypothesis: str,
    question_type: str,
    question_id: str,
) -> str:
    """Select and fill the per-type correctness prompt.

    ``answer`` carries the gold answer for most types, the rubric for
    ``single-session-preference``, and the explanation for abstention
    questions -- mirroring how the dataset overloads the field and how the
    official judge consumes it.
    """
    if is_abstention(question_id, question_type):
        return _ABSTENTION.format(question, answer, hypothesis)

    base_type = question_type.removesuffix("_abs")
    if base_type == "temporal-reasoning":
        return _TEMPORAL.format(question, answer, hypothesis)
    if base_type == "knowledge-update":
        return _KNOWLEDGE_UPDATE.format(question, answer, hypothesis)
    if base_type == "single-session-preference":
        return _PREFERENCE.format(question, answer, hypothesis)
    # single-session-user / single-session-assistant / multi-session and
    # any unrecognised type fall through to the base prompt, matching the
    # official script's default.
    return _BASE.format(question, answer, hypothesis)


class Judge:
    """litellm-backed correctness judge (frontier remote model by default)."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_base: str | None = None,
        *,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_base = api_base
        self.timeout = timeout

    def judge(
        self,
        *,
        question: str,
        answer: str,
        hypothesis: str,
        question_type: str,
        question_id: str,
    ) -> bool | None:
        """Return True/False for correct/incorrect, or None on judge error."""
        prompt = build_judge_prompt(
            question=question,
            answer=answer,
            hypothesis=hypothesis,
            question_type=question_type,
            question_id=question_id,
        )
        # A non-reasoning remote judge answers yes/no within max_tokens=10 (the
        # official setting). A local reasoning judge (ollama, e.g. Qwen3) emits
        # hidden reasoning that the /no_think prompt switch does NOT disable, so
        # max_tokens=10 returns an empty string -> always-False. The native
        # think=False option actually disables it; the verdict is then a prompt
        # yes/no.
        max_tokens = 10
        extra: dict[str, object] = {}
        if self.model.startswith("ollama/"):
            extra["think"] = False
            max_tokens = 16
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=self.timeout,
                **extra,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Error during judge evaluation: %s", e)
            return None
        return "yes" in content.strip().lower()
