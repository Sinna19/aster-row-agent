"""
Order lookup tool.

This is a plain deterministic function, not an LLM call. It is the only
thing allowed to touch orders.json, and it never returns internal/PII
fields to the caller -- so even a fully compromised prompt cannot leak
them, because they're simply not present in the returned object.

Status-derived business rules (stale ETA suppression, missing-ETA wording,
exception handoff) are implemented here in code rather than left to the
model, per the assignment's requirement that these be handled reliably.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

ORDER_ID_RE = re.compile(r"^ORD-\d+$")

# Fields explicitly allow-listed as customer-safe by data/orders-data-dictionary.md
_SAFE_ITEM_FIELDS = ("name", "quantity", "final_sale")


@dataclass
class OrderLookupResult:
    found: bool
    order_id: str
    status: Optional[str] = None
    membership_tier: Optional[str] = None
    items: Optional[list] = None
    placed_at: Optional[str] = None
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    # Guidance flags computed here so the model doesn't have to infer them:
    eta_suppressed_reason: Optional[str] = None  # why an ETA field should not be quoted
    requires_handoff: bool = False
    handoff_reason: Optional[str] = None
    error: Optional[str] = None

    def to_safe_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("found", "order_id")}


class OrderLookupTool:
    def __init__(self, orders_path: str):
        with open(orders_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.snapshot_at = data["snapshot_at"]
        self._by_id = {o["order_id"]: o for o in data["orders"]}

    @staticmethod
    def normalize_order_id(raw: str) -> str:
        """Normalize harmless differences: whitespace, case, stray punctuation."""
        cleaned = raw.strip().upper()
        cleaned = re.sub(r"[^A-Z0-9\-]", "", cleaned)
        return cleaned

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        order_id = self.normalize_order_id(raw_order_id)

        if not ORDER_ID_RE.match(order_id):
            # Fallback: the caller (model) may have passed a whole phrase
            # instead of an isolated ID, e.g. "check ORD-1007 please" or
            # "order 1007". Try to recover a well-formed order ID embedded
            # in the text before giving up. This does NOT guess a different
            # order -- it only extracts an ID that is already present.
            m = re.search(r"ORD-?\d+", raw_order_id.upper())
            if m:
                candidate = re.sub(r"ORD-?", "ORD-", m.group(0))
                if ORDER_ID_RE.match(candidate):
                    order_id = candidate

        if not ORDER_ID_RE.match(order_id):
            return OrderLookupResult(
                found=False,
                order_id=raw_order_id,
                error="malformed_order_id",
                requires_handoff=False,
            )

        order = self._by_id.get(order_id)
        if order is None:
            return OrderLookupResult(
                found=False,
                order_id=order_id,
                error="not_found",
                requires_handoff=True,
                handoff_reason="Order ID was not found. The customer should double-check the ID or contact support.",
            )

        status = order["status"]
        items = [{k: it[k] for k in _SAFE_ITEM_FIELDS} for it in order.get("items", [])]

        result = OrderLookupResult(
            found=True,
            order_id=order_id,
            status=status,
            membership_tier=order.get("membership_tier"),
            items=items,
            placed_at=order.get("placed_at"),
            status_updated_at=order.get("status_updated_at"),
            carrier=order.get("carrier"),
            tracking_number=order.get("tracking_number"),
            customer_safe_message=order.get("customer_safe_message"),
        )

        # Status-derived rules -- computed here, not left to the model.
        if status in ("cancelled", "returned"):
            # Stale shipped_at/delivered_at/estimated_delivery must not be
            # presented as if the order were still moving.
            result.shipped_at = None
            result.delivered_at = None
            result.estimated_delivery = None
            result.eta_suppressed_reason = (
                f"Order is {status}; any prior delivery estimate is stale and must not be reported."
            )
        else:
            result.shipped_at = order.get("shipped_at")
            result.delivered_at = order.get("delivered_at")
            result.estimated_delivery = order.get("estimated_delivery")
            if status == "shipped" and not result.estimated_delivery:
                result.eta_suppressed_reason = (
                    "Order has shipped but no delivery estimate is available; "
                    "do not calculate or invent one."
                )

        if status == "exception":
            result.requires_handoff = True
            result.handoff_reason = (
                "Order is in an exception state and requires support review."
            )

        return result


if __name__ == "__main__":
    import os

    tool = OrderLookupTool(os.path.join(os.path.dirname(__file__), "..", "..", "data", "orders.json"))
    for oid in ["ord-1007 ", "ORD-1004", "ORD-1011", "ORD-9999", "not-an-id", "ORD-1010"]:
        r = tool.lookup(oid)
        print(oid, "->", r.to_safe_dict())
