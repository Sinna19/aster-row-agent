import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluate import evaluate_expectations


def all_pass(checks):
    return all(c["pass"] for c in checks)


def test_must_include_and_must_not_include():
    expect = {"must_include": ["30 calendar days"], "must_not_include": ["60 days"]}
    good = evaluate_expectations(expect, "You have 30 calendar days from delivery.", [], False)
    assert all_pass(good)

    bad = evaluate_expectations(expect, "You have 60 days from delivery.", [], False)
    assert not all_pass(bad)


def test_narrow_no_break_space_does_not_break_exact_match():
    # Verified live failure mode: some models render "August 22, 2026" using
    # U+202F (narrow no-break space) instead of a plain space, which is
    # pixel-identical everywhere but fails a naive substring check.
    expect = {"must_include": ["August 22, 2026"]}
    text = "It will arrive on **August\u202f22,\u202f2026** per the carrier."
    result = evaluate_expectations(expect, text, [], False)
    assert all_pass(result)


def test_required_and_forbidden_sources():
    expect = {"required_sources": ["01-returns-policy-current.md"], "forbidden_sources_as_authority": ["02-returns-policy-legacy.md"]}
    good = evaluate_expectations(expect, "Per 01-returns-policy-current.md, 30 days.", [], False)
    assert all_pass(good)

    bad = evaluate_expectations(expect, "Per 02-returns-policy-legacy.md, 45 days.", [], False)
    assert not all_pass(bad)


def test_tool_not_called_detects_a_call():
    expect = {"tool": "not_called"}
    good = evaluate_expectations(expect, "some answer", [], False)
    assert all_pass(good)

    bad = evaluate_expectations(expect, "some answer", [{"tool": "order_lookup", "arguments": {"order_id": "ORD-1"}}], False)
    assert not all_pass(bad)


def test_tool_order_lookup_argument_match():
    expect = {"tool": "order_lookup", "tool_arguments": {"order_id": "ORD-1007"}}
    calls = [{"tool": "order_lookup", "arguments": {"order_id": "ord-1007 "}, "result": {}}]
    result = evaluate_expectations(expect, "shipped", calls, False)
    assert all_pass(result)


def test_must_not_invent_status_when_tool_not_called():
    expect = {"must_not_invent": ["order status"]}
    good = evaluate_expectations(expect, "Can you share your order ID?", [], False)
    assert all_pass(good)

    bad = evaluate_expectations(expect, "Your order is shipped and on its way!", [], False)
    assert not all_pass(bad)


def test_must_ask_for_accepts_order_number_synonym():
    expect = {"must_ask_for": ["order ID"]}
    result = evaluate_expectations(expect, "Could you share your order number?", [], False)
    assert all_pass(result)


def test_generic_signoff_does_not_trigger_false_handoff():
    expect = {"handoff": False}
    text = "Your order shipped. Let me know if you need anything else, or feel free to contact support."
    result = evaluate_expectations(expect, text, [], False)
    assert all_pass(result), "generic boilerplate sign-off should not be read as a real handoff signal"


def test_specific_specialist_language_does_trigger_handoff():
    expect = {"handoff": True}
    text = "I can't approve this myself, but I can connect you with a specialist who can."
    result = evaluate_expectations(expect, text, [], False)
    assert all_pass(result)


def test_handoff_regex_tolerates_word_insertions():
    # Real live failure: "a human specialist" (old literal phrase) didn't
    # match "a human support specialist" (what the model actually wrote).
    expect = {"handoff": True}
    text = "A human support specialist can review this and confirm."
    result = evaluate_expectations(expect, text, [], False)
    assert all_pass(result)
    expect = {"handoff": True}
    assert all_pass(evaluate_expectations(expect, "text", [], True))
    assert not all_pass(evaluate_expectations(expect, "text", [], False))


def test_conflict_requires_both_sides_present():
    expect = {"must_not_silently_choose_one": True}
    both = evaluate_expectations(
        expect,
        "Sources disagree: one says hand-wash the body, another says all components are dishwasher safe.",
        [],
        True,
    )
    assert all_pass(both)

    one_side_only = evaluate_expectations(expect, "It's dishwasher safe, go ahead.", [], False)
    assert not all_pass(one_side_only)
