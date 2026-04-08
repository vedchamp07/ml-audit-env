#!/usr/bin/env python3
"""
ML Experiment Integrity Auditor - Baseline Inference Script v4.0

Competition-compatible baseline using OpenAI client.
Uses agentic LLM calls at every step (no dumb fallback).
Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN (or OPENAI_API_KEY).
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv

# -- Configuration -----------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# IMPORTANT: Evaluators often parse stdout. Keep stdout restricted to:
#   [START] ...
#   [STEP]  ...
#   [END]   ...
# Route any diagnostics to stderr.
DEBUG = os.getenv("DEBUG", "0") == "1"


def _dprint(msg: str) -> None:
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)

# Competition-required env vars (with backward-compatible fallbacks)
API_BASE_URL = (
    os.environ.get("API_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://router.huggingface.co/v1"
).strip().strip('"').strip("'")

# Prioritize OPENAI_API_KEY when using OpenAI API
if "openai" in API_BASE_URL.lower():
    API_KEY = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
        or ""
    ).strip().strip('"').strip("'")
else:
    API_KEY = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    ).strip().strip('"').strip("'")

MODEL_NAME = (os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct").strip().strip('"').strip("'")

ENV_URL = (
    os.environ.get("ENV_URL") or "http://localhost:7860"
).strip().strip('"').strip("'")

# Sample-style metadata
BENCHMARK = os.getenv("BENCHMARK", "ml-audit-bench")
TASK_NAME = os.getenv("TASK_NAME") or os.getenv("TASK_FILTER") or "easy"

TEMPERATURE = 0.0
MAX_TOKENS = 800
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
REQUEST_TIMEOUT = 30
RETRY_DELAYS = [2, 4, 8]
TASK_FILTER = os.getenv("TASK_FILTER", None)
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "3"))

SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

ARTIFACT_TRUNCATION = {
    "split_config": 4000,
    "preprocessing": 2500,
    "experiment_notes": 2000,
    "validation_strategy": 2000,
    "eval_report": 2000,
    "_default": 1500,
}

COMPARE_HINTS = [
    {
        "artifacts": {"run_history", "experiment_notes"},
        "hint": (
            "You have now inspected both run_history and experiment_notes. "
            "Consider using compare(artifact_a='run_history', artifact_b='experiment_notes') "
            "to check whether the total number of runs disclosed in experiment_notes "
            "matches the actual run count in run_history (V5: undisclosed multiple runs)."
        ),
        "violation": "V5",
    },
    {
        "artifacts": {"validation_strategy", "eval_report"},
        "hint": (
            "You have now inspected both validation_strategy and eval_report. "
            "Consider using compare(artifact_a='validation_strategy', artifact_b='eval_report') "
            "to check whether all metrics_tracked in validation_strategy are reported in "
            "eval_report (V6: selective metric reporting)."
        ),
        "violation": "V6",
    },
]

try:
    SEED = int((os.environ.get("SEED") or "42").strip().strip('"').strip("'"))
except ValueError:
    SEED = 42

# DRY_RUN mode: skip LLM calls and use deterministic action sequence
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"


def get_dry_run_action(obs: dict, step_num: int) -> dict:
    """Return deterministic action for DRY_RUN mode (no LLM calls)."""
    inspected = set(obs.get("inspected_artifacts", []))
    available = obs.get("available_artifacts", [])
    remaining = [a for a in available if a not in inspected]

    # Deterministic inspection order
    priority = ["preprocessing", "split_config", "model_config", "dataset_info",
                "validation_strategy", "eval_report", "run_history", "experiment_notes"]

    for artifact in priority:
        if artifact in remaining:
            return {"type": "inspect", "artifact": artifact}

    # If all inspected or step limit approaching, submit
    return {"type": "submit", "verdict": "pass", "summary": "dry run complete"}


# -- System Prompt -----------------------------------------------------------
# CRITICAL FIX: This prompt teaches the agent HOW violations actually manifest
# in the artifacts.  Previous versions pointed at JSON keys the injectors
# deliberately remove (fit_scope, overlap_count, entity_overlap_count).

DEFAULT_SYSTEM_PROMPT = """\
You are an ML experiment integrity auditor. Output only ONE JSON action object.

Hard constraints:
- Never output prose, markdown, or arrays.
- Never flag without evidence_quote copied exactly from artifact text.
- Never submit until all required checks complete.

Action JSON must be one of:
{"type":"inspect","artifact":"<name>"}
{"type":"compare","artifact_a":"<name>","artifact_b":"<name>"}
{"type":"flag","violation_type":"V1","evidence_artifact":"<name>","evidence_quote":"<exact>","severity":"high"}
{"type":"unflag","flag_id":"<id>"}
{"type":"submit","verdict":"pass|revise|reject","summary":"<findings>"}

Violation checklist:
- V1 preprocessing fit_transform done before train_test_split.
- V2 dataset_type=timeseries and split_config.shuffle=true.
- V3 target_column EXACT match appears in model_config.feature_columns.
- V4 any ID overlap between train_ids_sample and test_ids_sample.
- V5 run_history.total_runs>1 and notes fail to disclose multi-run or run count.
    Quote EXACT JSON evidence from run_history, e.g. "\"total_runs\": 15".
    Do NOT paraphrase as "15 runs".
- V6 metrics_tracked count >=3 and reported_metrics key count ==1.
    Quote EXACT JSON evidence from eval_report, e.g. "\"reported_metrics\": [\"accuracy\"]".
- V7 any overlap between train_entities_sample and test_entities_sample.
- V8 validation strategy indicates tuning on test set.

Execution protocol:
Step A: inspect dataset_info
Step B: inspect preprocessing
Step C: inspect split_config
Step D: inspect model_config
Step E: if available, inspect validation_strategy
Step F: inspect eval_report
Step G: if available, inspect run_history
Step H: inspect experiment_notes
Step I: if both are available, compare validation_strategy + eval_report to verify V6
Step J: if both are available, compare run_history + experiment_notes to verify V5
Step J: flag each confirmed violation once
Step K: submit final verdict

== MANDATORY compare() FOR V5/V6 ==
If run_history and experiment_notes are both inspected and available, next action must be:
{"type":"compare","artifact_a":"run_history","artifact_b":"experiment_notes"}
If validation_strategy and eval_report are both inspected and available, next action must be:
{"type":"compare","artifact_a":"validation_strategy","artifact_b":"eval_report"}
After a compare pair has been executed once, do not repeat the same compare again unless new related artifacts were inspected.

== CRITICAL: PATTERNS THAT LOOK SUSPICIOUS BUT ARE NOT VIOLATIONS ==
1) GroupShuffleSplit or StratifiedGroupKFold with correct group/entity column is NOT V7.
2) Very high accuracy can be legitimate if dataset_info explains class separation.
3) Early overfitting that recovers by later epochs is NOT a violation.
4) Small test_size can be justified for rare conditions; check notes before flagging.
5) Unusual learning_rate can be valid for optimizer choice; check model_config context.
6) Hyperparameter tuning on cross-validation folds is NOT V8.
CRITICAL V1 false-positive guard: if preprocessing splits first and then uses fit/fit_transform on X_train with transform on X_test, that is correct and NOT V1.

If uncertain, do not flag immediately. False positives (-0.10) are costly.

== WHEN TO SUBMIT ==
Do not submit before checking core artifacts (dataset_info, preprocessing, split_config)
and completing available cross-check compares for V5/V6.
Under budget pressure, prioritize compare+flag over extra re-inspections.

Anti-confusion rules:
- Exact match required for V3; partial string does not count.
- For V4 and V7, manually compare arrays; do not rely on helper counts.
- GroupShuffleSplit is not a violation.
- If uncertain, inspect more artifacts instead of flagging.
"""

SYSTEM_PROMPT = (os.environ.get("SYSTEM_PROMPT") or "").strip()
SYSTEM_PROMPT_FILE = (os.environ.get("SYSTEM_PROMPT_FILE") or "").strip().strip('"').strip("'")

if not SYSTEM_PROMPT and SYSTEM_PROMPT_FILE:
    try:
        SYSTEM_PROMPT = Path(SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
    except OSError as exc:
        _dprint(f"WARN: Failed to read SYSTEM_PROMPT_FILE ({SYSTEM_PROMPT_FILE}): {exc}")

if not SYSTEM_PROMPT:
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


# -- HTTP helpers ------------------------------------------------------------

def _http(method, endpoint, **kwargs):
    url = f"{ENV_URL}{endpoint}"
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            resp = (requests.get if method == "GET" else requests.post)(url, **kwargs)
            return resp
        except requests.RequestException as exc:
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
            else:
                _dprint(f"ENV request failed: {exc}")
    return None


def env_request(method, endpoint, **kwargs):
    resp = _http(method, endpoint, **kwargs)
    if resp is None:
        return None
    try:
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        _dprint(f"ENV error: {exc}")
    return None


# -- LLM call ----------------------------------------------------------------

def llm_call(client, messages):
    use_mct = False
    include_temp = True
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            req = {"model": MODEL_NAME, "messages": messages, "stream": False, "timeout": REQUEST_TIMEOUT}
            if include_temp:
                req["temperature"] = TEMPERATURE
            if use_mct:
                req["max_completion_tokens"] = MAX_TOKENS
            else:
                req["max_tokens"] = MAX_TOKENS
            completion = client.chat.completions.create(**req)
            return completion.choices[0].message.content or ""
        except Exception as exc:
            err = str(exc)
            if "max_tokens" in err and "max_completion_tokens" in err:
                use_mct = True; continue
            if "temperature" in err and "default (1) value is supported" in err:
                include_temp = False; continue
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
            else:
                _dprint(f"LLM failed: {err[:200]}")
    return ""


def maybe_add_compare_hint(inspected_set, hint_configs, already_hinted):
    """Return hint string if both required artifacts are now inspected and hint not yet given."""
    for cfg in hint_configs:
        key = cfg["violation"]
        if key not in already_hinted and cfg["artifacts"].issubset(inspected_set):
            already_hinted.add(key)
            _dprint(f"Compare hint injected for {cfg['violation']} after inspecting {sorted(inspected_set)}")
            return cfg["hint"]
    return None


def _hint_to_compare_action(hint_text):
    if "run_history" in hint_text and "experiment_notes" in hint_text:
        return {"type": "compare", "artifact_a": "run_history", "artifact_b": "experiment_notes"}
    if "validation_strategy" in hint_text and "eval_report" in hint_text:
        return {"type": "compare", "artifact_a": "validation_strategy", "artifact_b": "eval_report"}
    return None


# -- Action parsing ----------------------------------------------------------

def parse_action(text):
    if not text:
        return None
    raw = text.strip()

    # Direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if "type" in obj:
                return obj
            if isinstance(obj.get("action"), dict) and "type" in obj["action"]:
                return obj["action"]
    except json.JSONDecodeError:
        pass

    # Code fences
    if "```" in raw:
        for i, chunk in enumerate(raw.split("```")):
            if i % 2 == 0:
                continue
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            try:
                obj = json.loads(c)
                if isinstance(obj, dict) and "type" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

    # Regex extract
    for match in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, re.DOTALL):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "type" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_action(action: dict) -> dict:
    """Map common off-spec action variants into valid OpenEnv action schema."""
    if not isinstance(action, dict):
        return action

    atype = str(action.get("type", "")).strip()

    # Common model aliases for inspect
    if atype in {"load_artifact", "read_artifact", "inspect_artifact"}:
        artifact = action.get("artifact") or action.get("name") or action.get("artifact_name") or "dataset_info"
        return {"type": "inspect", "artifact": artifact}

    # Common metadata request fallback -> inspect dataset_info
    if atype in {"request_experiment_metadata", "request_metadata"}:
        return {"type": "inspect", "artifact": "dataset_info"}

    # Field normalization for valid inspect actions
    if atype == "inspect" and "artifact" not in action:
        artifact = action.get("name") or action.get("artifact_name") or "dataset_info"
        return {"type": "inspect", "artifact": artifact}

    return action


# -- Context builder ---------------------------------------------------------

def truncate(val, max_len=None):
    """Truncate text to max_len characters with a compact overflow marker."""
    text = str(val)
    if max_len is None:
        max_len = ARTIFACT_TRUNCATION["_default"]
    return text if len(text) <= max_len else text[:max_len] + f"...[+{len(text)-max_len}]"


def build_context(obs, step_num, cache):
    flags = obs.get("flags_raised", []) or []
    flag_lines = [f"  {f.get('flag_id','?')}: {f.get('violation_type','?')} in {f.get('evidence_artifact','?')}"
                  for f in flags]

    excerpts = []
    for name in obs.get("inspected_artifacts", []):
        content = cache.get(name, "")
        if content:
            limit = ARTIFACT_TRUNCATION.get(name, ARTIFACT_TRUNCATION["_default"])
            excerpts.append(f"=== {name} ===\n{truncate(content, limit)}")

    return (
        f"Step {step_num+1}/{obs.get('step_budget', MAX_STEPS)}\n"
        f"experiment_id: {obs.get('experiment_id','?')}\n"
        f"dataset_type: {obs.get('dataset_type','?')}\n"
        f"available_artifacts: {obs.get('available_artifacts',[])}\n"
        f"inspected_artifacts: {obs.get('inspected_artifacts',[])}\n"
        f"flags ({len(flags)}):\n" + ("\n".join(flag_lines) if flag_lines else "  none") + "\n"
        f"last_action_result: {truncate(obs.get('last_action_result','N/A'))}\n"
        f"last_action_error: {obs.get('last_action_error') or 'none'}\n\n"
        "Cached artifacts:\n" + ("\n\n".join(excerpts) if excerpts else "  none yet") + "\n\n"
        "Return ONE JSON action."
    )


def build_state_summary(obs, step_num):
    """Build compact state summary injected before each LLM action call."""
    inspected = obs.get("inspected_artifacts", []) or []
    available = obs.get("available_artifacts", []) or []
    budget = int(obs.get("step_budget", MAX_STEPS))
    remaining_steps = max(0, budget - step_num)
    lines = [
        "State summary:",
        f"- inspected artifacts: {inspected}",
        f"- available artifacts: {available}",
        f"- remaining steps: {remaining_steps}",
    ]
    if remaining_steps <= 4:
        lines.append("- budget alert: <= 4 steps remaining; prioritize decisive checks and submission")
    return "\n".join(lines)


# -- Episode runner ----------------------------------------------------------

def _format_action_single_line(action: dict) -> str:
    """Format action as single-line JSON for [STEP] output."""
    return json.dumps(action, separators=(',', ':'))


def _clamp01(x: Any) -> float:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_clamp01(score):.3f} rewards={rewards_str}",
        flush=True,
    )


def run_episode(client, task, seed=SEED):
    """
    Run a single episode with competition-required [START]/[STEP]/[END] stdout format.
    """
    step_rewards: list[float] = []
    score = 0.0
    steps_taken = 0
    success = False
    response_text = ""  # ensure always defined

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset with seed, fallback to unseeded
        reset_data = env_request("POST", "/reset", params={"task": task, "seed": seed})
        if reset_data is None:
            reset_data = env_request("POST", "/reset", params={"task": task})
        if reset_data is None or not isinstance(reset_data, dict):
            _dprint("Failed to reset or received invalid response")
            return 0.0

        obs = reset_data.get("observation", reset_data)
        if not isinstance(obs, dict):
            _dprint("Invalid observation format")
            return 0.0
        
        cache = {}
        history = []
        invalid_streak = 0
        already_hinted = set()
        pending_hints = []
        pending_compare_overrides = []

        budget = int(obs.get("step_budget", MAX_STEPS))
        budget = max(1, min(budget, MAX_STEPS))

        for step_num in range(1, budget + 1):
            ctx = build_context(obs, step_num - 1, cache)
            state_summary = build_state_summary(obs, step_num - 1)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for act_msg, res_msg in history[-2:]:
                messages.append({"role": "assistant", "content": act_msg})
                messages.append({"role": "user", "content": res_msg})
            for hint in pending_hints:
                messages.append({"role": "user", "content": hint})
            pending_hints.clear()
            messages.append({"role": "user", "content": state_summary})
            messages.append({"role": "user", "content": ctx})

            # DRY_RUN mode: skip LLM, use deterministic actions
            response_text = ""
            if DRY_RUN:
                action = get_dry_run_action(obs, step_num)
            else:
                response_text = llm_call(client, messages)
                action = parse_action(response_text)

            if action is not None:
                action = normalize_action(action)

            if action is None:
                invalid_streak += 1
                _dprint(f"Invalid JSON ({invalid_streak}): {response_text[:120]}")
                available = obs.get("available_artifacts", [])
                inspected = set(obs.get("inspected_artifacts", []))
                remaining = [a for a in available if a not in inspected]
                if remaining:
                    action = {"type": "inspect", "artifact": remaining[0]}
                else:
                    fallback_artifact = available[0] if available else "dataset_info"
                    action = {"type": "inspect", "artifact": fallback_artifact}
            else:
                invalid_streak = 0

            if action is not None and pending_compare_overrides and action.get("type") != "compare":
                forced = pending_compare_overrides.pop(0)
                if forced:
                    action = forced

            # Prevent duplicate flags
            if action.get("type") == "flag":
                existing = obs.get("flags_raised", []) or []
                dup = any(
                    f.get("violation_type") == action.get("violation_type")
                    and f.get("evidence_artifact") == action.get("evidence_artifact")
                    for f in existing
                )
                if dup:
                    available = obs.get("available_artifacts", [])
                    inspected = set(obs.get("inspected_artifacts", []))
                    remaining = [a for a in available if a not in inspected]
                    if remaining:
                        action = {"type": "inspect", "artifact": remaining[0]}
                    else:
                        action = {"type": "submit", "verdict": "reject",
                                  "summary": "Found violations; submitting"}

            # Prevent premature submit before core checks/cross-checks are completed.
            if action.get("type") == "submit":
                inspected = set(obs.get("inspected_artifacts", []))
                available = set(obs.get("available_artifacts", []))
                core_needed = ["dataset_info", "preprocessing", "split_config"]
                missing_core = [a for a in core_needed if a in available and a not in inspected]
                if missing_core:
                    action = {"type": "inspect", "artifact": missing_core[0]}
                elif (
                    {"run_history", "experiment_notes"}.issubset(available)
                    and {"run_history", "experiment_notes"}.issubset(inspected)
                    and step_num < MAX_STEPS - 2
                ):
                    action = {"type": "compare", "artifact_a": "run_history", "artifact_b": "experiment_notes"}
                elif (
                    {"validation_strategy", "eval_report"}.issubset(available)
                    and {"validation_strategy", "eval_report"}.issubset(inspected)
                    and step_num < MAX_STEPS - 2
                ):
                    action = {"type": "compare", "artifact_a": "validation_strategy", "artifact_b": "eval_report"}

            # Budget guard: end cleanly on final allowed step.
            if step_num == budget and action.get("type") != "submit":
                action = {"type": "submit", "verdict": "pass", "summary": "budget_end"}

            result = env_request("POST", "/step", json={"action": action})
            if result is None:
                result = env_request("POST", "/step",
                    json={"action": {"type": "submit", "verdict": "reject", "summary": "env_error"}})
                if result is None or not isinstance(result, dict):
                    _dprint("Step request failed")
                    return 0.0

            obs = result.get("observation", result)
            if not isinstance(obs, dict):
                _dprint("Invalid observation in step response")
                return 0.0
            
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error_str = obs.get("last_action_error")
            steps_taken = step_num
            try:
                step_rewards.append(float(reward or 0.0))
            except (TypeError, ValueError):
                step_rewards.append(0.0)

            # [STEP] line - emitted after each env.step()
            action_str = _format_action_single_line(action)
            try:
                reward_f = float(reward or 0.0)
            except (TypeError, ValueError):
                reward_f = 0.0
            log_step(step=steps_taken, action=action_str, reward=reward_f, done=bool(done), error=error_str)

            # Cache artifacts for evidence quoting
            if action.get("type") == "inspect" and not obs.get("last_action_error"):
                art = str(action.get("artifact", ""))
                if art:
                    cache[art] = str(obs.get("last_action_result", ""))
            elif action.get("type") == "compare" and not obs.get("last_action_error"):
                text = str(obs.get("last_action_result", ""))
                for chunk in re.split(r"\n=== ", text):
                    c = chunk.strip()
                    if c.startswith("=== "): c = c[4:]
                    if " ===\n" in c:
                        n, b = c.split(" ===\n", 1)
                        cache[n.strip()] = b

            history.append((json.dumps(action), truncate(obs.get("last_action_result", ""), 400)))

            inspected_set = set(obs.get("inspected_artifacts", []) or [])
            compare_hint = maybe_add_compare_hint(inspected_set, COMPARE_HINTS, already_hinted)
            if compare_hint:
                pending_hints.append(compare_hint)
                pending_compare_overrides.append(_hint_to_compare_action(compare_hint))

            if done:
                try:
                    score = _clamp01(result.get("info", {}).get("score", 0.0)) if isinstance(result, dict) else 0.0
                except (ValueError, TypeError, AttributeError):
                    score = 0.0
                success = score >= SUCCESS_SCORE_THRESHOLD
                return float(score)

        # If we exhausted budget without a done=True response, we still emit [END]
        # with score=0.0 (conservative) and success=false.
        return float(score)

    finally:
        # Always attempt to close server-side episode (best-effort)
        try:
            env_request("POST", "/close")
        except Exception as exc:
            _dprint(f"/close failed: {exc}")
        log_end(success=success, steps=steps_taken, score=score, rewards=step_rewards)


# -- Main --------------------------------------------------------------------

def main():
    # Keep stdout limited to the required [START]/[STEP]/[END] lines.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    run_episode(client, TASK_NAME, seed=SEED)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Script interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] Script crashed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)