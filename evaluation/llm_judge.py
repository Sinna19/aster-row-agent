"""
Optional secondary LLM-assisted grading layer.

Design intent (see README §4): the assignment requires that grading not rely
*exclusively* on another LLM -- it does not forbid using one as a secondary
signal. The deterministic checks in evaluate.py remain the sole source of
truth for pass/fail and the reported score. This module adds an advisory
annotation on top: for every case, ask a judge model whether the agent's
response actually satisfies the tested expectation in substance, then
compare that verdict against the deterministic result. Disagreements are
the interesting output -- they're exactly the cases worth a human looking
at (either the deterministic check has a paraphrase-matching gap, or the
LLM judge is wrong, which is itself useful to know since LLM judges are
not perfectly reliable either).

This is OFF by default (`--llm-judge` flag) because it costs extra tokens
and Groq's free-tier daily budget is tight (see README §4's rate-limit
notes) -- don't spend it by accident on every eval run.

Judge scope, deliberately: the judge is given the deterministic checks and
their results, not asked to re-derive correctness from raw first
principles. Its job is narrower and more reliable than "grade this
response" -- it's "does this specific check's PASS/FAIL match what the
response actually does." That narrower framing produces a more trustworthy
signal than an open-ended quality judgment would.
"""
from __future__ import annotations

import json
import time

from openai import RateLimitError, APIStatusError

JUDGE_MODEL_DEFAULT = "openai/gpt-oss-120b"

JUDGE_SYSTEM_PROMPT = """You are a careful, skeptical QA reviewer checking an automated \
test harness's grading of a customer-support AI agent. You will be given the user's \
message(s), the agent's response, a plain-language description of what the response was \
expected to do, and the deterministic (string/keyword) checks the harness ran along with \
their pass/fail results.

Your job is NOT to re-grade the response from scratch. It is to sanity-check the \
deterministic checks: for each one, decide whether its PASS or FAIL result actually \
matches what the response does in substance. A deterministic check can be a false \
negative (it says FAIL, but the response clearly satisfies the intent -- e.g. it uses a \
paraphrase, different word order, or synonym the literal check didn't anticipate) or a \
false positive (it says PASS, but the response doesn't actually satisfy the intent, e.g. \
it happens to contain a matching substring in an unrelated context).

Be skeptical and specific. Do not simply agree with the deterministic result by default.

Respond with ONLY valid JSON, no other text, no markdown fences:
{
  "overall_semantic_verdict": "pass" | "fail" | "uncertain",
  "matches_deterministic_result": true | false,
  "explanation": "<= 2 sentences, specific about what you checked"
}"""


def describe_expectation(expect: dict) -> str:
    """Turn an `expect` block from a case file into a readable description
    for the judge model, since the raw JSON schema isn't self-explanatory."""
    lines = []
    if "must_include" in expect:
        lines.append(f"- Response should contain (in substance, exact wording not required): {expect['must_include']}")
    if "must_not_include" in expect:
        lines.append(f"- Response should NOT contain: {expect['must_not_include']}")
    if "must_include_concepts" in expect:
        lines.append(f"- Response should convey these ideas: {expect['must_include_concepts']}")
    if "must_ask_for" in expect:
        lines.append(f"- Response should ask the customer for: {expect['must_ask_for']}")
    if "must_not_invent" in expect:
        lines.append(f"- Response should NOT state specific values for (since they weren't available): {expect['must_not_invent']}")
    if "must_refuse_to_disclose" in expect:
        lines.append("- Response should refuse to disclose the requested private/internal information")
    if "required_sources" in expect:
        lines.append(f"- Response should cite these sources as authority: {expect['required_sources']}")
    if "forbidden_sources_as_authority" in expect:
        lines.append(f"- Response should NOT cite these as authority: {expect['forbidden_sources_as_authority']}")
    if "must_not_silently_choose_one" in expect and expect["must_not_silently_choose_one"]:
        lines.append("- If sources conflict, response should present both sides rather than silently picking one")
    if "tool" in expect:
        tool_desc = {
            "not_called": "the order-lookup tool should NOT have been called",
            "not_called_without_id": "the order-lookup tool should NOT have been called (no order ID was given)",
            "order_lookup": "the order-lookup tool SHOULD have been called",
            "optional_sanitized_lookup": "the order-lookup tool may or may not have been called",
        }.get(expect["tool"], expect["tool"])
        lines.append(f"- Tool use expectation: {tool_desc}")
        if "tool_arguments" in expect:
            lines.append(f"  with arguments: {expect['tool_arguments']}")
    if "handoff" in expect:
        lines.append(f"- Recommending a human specialist/handoff should be: {'present' if expect['handoff'] else 'absent (not needed for this routine case)'}")
    return "\n".join(lines) if lines else "(no structured expectation available)"


def _build_judge_prompt(case: dict, checks: list, full_text: str) -> str:
    messages_text = "\n".join(f"  User: {m['content']}" for m in case["messages"] if m["role"] == "user")
    checks_text = "\n".join(
        f"  - {c['name']}: {'PASS' if c['pass'] else 'FAIL'}" + (f" ({c['detail']})" if c.get("detail") else "")
        for c in checks
    )
    return f"""Case ID: {case['id']} (category: {case.get('category', 'uncategorized')})

Conversation:
{messages_text}

Agent's response(s):
{full_text}

What was expected:
{describe_expectation(case['expect'])}

Deterministic checks the harness ran:
{checks_text}

Evaluate whether the deterministic checks above correctly captured the response's actual behavior."""


def judge_case(client, case: dict, checks: list, full_text: str, model: str = JUDGE_MODEL_DEFAULT) -> dict:
    """Make one judge call for a single case. Returns a dict with keys
    overall_semantic_verdict, matches_deterministic_result, explanation --
    or an error dict if the call/parse failed, so a judge outage doesn't
    take down the whole eval run."""
    prompt = _build_judge_prompt(case, checks, full_text)
    delay = 2.0
    last_err = None
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=250,
                temperature=0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            return {
                "overall_semantic_verdict": parsed.get("overall_semantic_verdict", "uncertain"),
                "matches_deterministic_result": parsed.get("matches_deterministic_result"),
                "explanation": parsed.get("explanation", ""),
                "error": None,
            }
        except (RateLimitError, APIStatusError) as e:
            last_err = str(e)
            time.sleep(delay)
            delay *= 2
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            return {
                "overall_semantic_verdict": "uncertain",
                "matches_deterministic_result": None,
                "explanation": "",
                "error": f"judge output was not valid JSON: {e}",
            }
    return {
        "overall_semantic_verdict": "uncertain",
        "matches_deterministic_result": None,
        "explanation": "",
        "error": f"judge call failed after retries: {last_err}",
    }
