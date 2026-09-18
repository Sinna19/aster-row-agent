import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.llm_judge import describe_expectation, judge_case


def test_describe_expectation_covers_common_fields():
    expect = {
        "must_include": ["30 calendar days"],
        "required_sources": ["01-returns-policy-current.md"],
        "tool": "order_lookup",
        "tool_arguments": {"order_id": "ORD-1007"},
        "handoff": False,
    }
    desc = describe_expectation(expect)
    assert "30 calendar days" in desc
    assert "01-returns-policy-current.md" in desc
    assert "SHOULD have been called" in desc
    assert "absent" in desc


def test_describe_expectation_empty_expect_does_not_crash():
    assert describe_expectation({}) == "(no structured expectation available)"


def _fake_case():
    return {
        "id": "test-case",
        "category": "retrieval",
        "messages": [{"role": "user", "content": "How long can I return a bag?"}],
        "expect": {"must_include": ["30 calendar days"]},
    }


class _FakeCompletions:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        msg = SimpleNamespace(content=self.response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeClient:
    def __init__(self, response_text):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response_text))


def test_judge_case_parses_valid_json():
    verdict_json = json.dumps({
        "overall_semantic_verdict": "pass",
        "matches_deterministic_result": True,
        "explanation": "Response clearly states the 30-day window.",
    })
    client = _FakeClient(verdict_json)
    checks = [{"name": "must_include:30 calendar days", "pass": True, "detail": ""}]
    result = judge_case(client, _fake_case(), checks, "You have 30 calendar days to return.")
    assert result["overall_semantic_verdict"] == "pass"
    assert result["matches_deterministic_result"] is True
    assert result["error"] is None


def test_judge_case_strips_markdown_fences():
    verdict_json = "```json\n" + json.dumps({
        "overall_semantic_verdict": "fail",
        "matches_deterministic_result": False,
        "explanation": "Response uses different wording than expected.",
    }) + "\n```"
    client = _FakeClient(verdict_json)
    checks = [{"name": "must_include:45 calendar days", "pass": False, "detail": ""}]
    result = judge_case(client, _fake_case(), checks, "45-calendar-day window applies.")
    assert result["matches_deterministic_result"] is False
    assert result["error"] is None


def test_judge_case_handles_malformed_json_gracefully():
    client = _FakeClient("this is not json at all")
    checks = [{"name": "must_include:x", "pass": True, "detail": ""}]
    result = judge_case(client, _fake_case(), checks, "some response")
    assert result["overall_semantic_verdict"] == "uncertain"
    assert result["error"] is not None
