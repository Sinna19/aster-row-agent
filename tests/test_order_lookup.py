import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.order_lookup import OrderLookupTool

ORDERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")


def get_tool():
    return OrderLookupTool(ORDERS_PATH)


def test_normalizes_lowercase_and_whitespace():
    tool = get_tool()
    r = tool.lookup("  ord-1007  ")
    assert r.found
    assert r.order_id == "ORD-1007"


def test_unknown_order_id_triggers_handoff_and_no_status():
    tool = get_tool()
    r = tool.lookup("ORD-9999")
    assert not r.found
    assert r.requires_handoff
    d = r.to_safe_dict()
    assert "status" not in d or d.get("status") is None


def test_malformed_order_id_handled_safely():
    tool = get_tool()
    r = tool.lookup("banana")
    assert not r.found
    assert r.error == "malformed_order_id"


def test_cancelled_order_suppresses_stale_eta():
    tool = get_tool()
    r = tool.lookup("ORD-1004")
    assert r.status == "cancelled"
    assert r.estimated_delivery is None
    assert r.eta_suppressed_reason is not None


def test_returned_order_suppresses_stale_eta():
    tool = get_tool()
    r = tool.lookup("ORD-1008")
    assert r.status == "returned"
    assert r.estimated_delivery is None


def test_shipped_without_eta_flagged_not_invented():
    tool = get_tool()
    r = tool.lookup("ORD-1011")
    assert r.status == "shipped"
    assert r.estimated_delivery is None
    assert "do not" in r.eta_suppressed_reason.lower() or "unavailable" in r.eta_suppressed_reason.lower()


def test_exception_status_requires_handoff():
    tool = get_tool()
    r = tool.lookup("ORD-1010")
    assert r.status == "exception"
    assert r.requires_handoff


def test_no_pii_or_internal_fields_ever_returned():
    tool = get_tool()
    forbidden_keys = {"customer", "internal", "email", "shipping_address", "risk_score", "warehouse_note", "support_tags", "name"}
    for oid in ["ORD-1001", "ORD-1007", "ORD-1010"]:
        d = get_tool().lookup(oid).to_safe_dict()
        assert forbidden_keys.isdisjoint(d.keys())
        # also check nested item dicts don't smuggle anything extra
        for item in d.get("items", []):
            assert set(item.keys()) <= {"name", "quantity", "final_sale"}


def test_recovers_order_id_embedded_in_a_phrase():
    tool = get_tool()
    r = tool.lookup("check ORD-1007 please")
    assert r.found
    assert r.order_id == "ORD-1007"


def test_does_not_guess_an_id_when_prefix_is_missing():
    tool = get_tool()
    r = tool.lookup("order 12345")
    assert not r.found
    assert r.error == "malformed_order_id"


def test_valid_order_matches_expected_fixture():
    tool = get_tool()
    r = tool.lookup("ORD-1007")
    assert r.status == "shipped"
    assert r.carrier == "UPS"
    assert r.estimated_delivery == "2026-08-22"
