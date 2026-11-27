"""Planning assistant: a Claude agent that answers questions by calling the app's own functions.

The model never invents numbers. It picks tools (KPIs, part search, scenarios, P&L,
suggested orders), the app runs them on the selected company's data, and the model
explains the results. Tools that would change the ERP only *draft* an action; a person
confirms it on the page before anything is written.

Setup: put ANTHROPIC_API_KEY (and optionally ASSISTANT_PASSWORD, ANTHROPIC_MODEL) in
.streamlit/secrets.toml locally or in the Streamlit Cloud app's Secrets.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_STEPS = 8            # tool round-trips per question
MAX_ROWS = 25            # rows returned to the model per table
MAX_TOKENS = 2000

SYSTEM = """You are the planning assistant inside a parts demand-planning dashboard for {company}.
You help planners, buyers and finance leaders understand inventory, forecasts, scenarios and the business case.

Rules:
- Get every number from a tool. Never estimate or invent figures. If no tool covers a question, say so.
- Data is {data_kind}. Say so when someone might mistake results for a real customer's.
- Be brief and concrete: lead with the answer and the key number, then up to 5 bullets. Use $ and % formatting.
- Name the page where the user can see more (Summary, Value impact, Scenario planning, Financial P&L,
  KPI scorecard, Forecast accuracy, Inventory, Part detail, Approvals, Mock ERP).
- Purchase orders and min/max changes: you may only DRAFT them with the draft tools. Tell the user to review
  and click Confirm below the chat. Never claim anything was written to the ERP.
- Service levels, order cycle and the special-order setting come from the sidebar; tell the user to change
  them there rather than assuming other values.
- If a question is outside inventory, forecasting, scenarios or the P&L, answer briefly or decline.
Today's sidebar settings: {settings}."""

TOOLS = [
    {"name": "company_overview",
     "description": "Headline figures for the selected company: inventory value, excess and dead stock, fill rate and "
                    "average inventory under today's ERP min/max vs. the recommended policy, lost sales, part counts, "
                    "and the annual dollar value of the software at the current sidebar settings.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "find_parts",
     "description": "Search and rank SKU-locations. Use for questions like 'which parts are most at risk', "
                    "'biggest excess', 'A parts at Detroit'. Returns up to `limit` rows.",
     "input_schema": {"type": "object", "properties": {
         "location_id": {"type": "string", "description": "Location id, e.g. DET. Omit for all."},
         "abc": {"type": "string", "enum": ["A", "B", "C"]},
         "pattern": {"type": "string", "enum": ["smooth", "erratic", "intermittent", "lumpy"]},
         "category": {"type": "string"},
         "supplier_id": {"type": "string"},
         "text": {"type": "string", "description": "Matches part number or description."},
         "sort_by": {"type": "string", "enum": ["stockout_risk", "excess_value", "annual_value", "lost_sales",
                                                "usage_value"],
                     "description": "stockout_risk = below reorder point, ranked by annual usage value."},
         "limit": {"type": "integer", "minimum": 1, "maximum": 25}}}},
    {"name": "part_detail",
     "description": "Everything about one part at one or all locations: pattern, ABC, cost, price, lead time, "
                    "on hand, on order, ERP min/max, recommended reorder point and order-up-to, recent demand.",
     "input_schema": {"type": "object", "properties": {
         "item_id": {"type": "string"}, "location_id": {"type": "string"}}, "required": ["item_id"]}},
    {"name": "list_values",
     "description": "Valid locations, categories and suppliers for the selected company. Call this before filtering "
                    "if you are unsure of an id or name.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "forecast_accuracy",
     "description": "Forecast KPIs (forecast accuracy = 1 - WAPE, WAPE, bias, MAPE) for our forecast vs. the ERP's "
                    "moving average over the 39-week backtest, optionally by segment.",
     "input_schema": {"type": "object", "properties": {
         "segment_by": {"type": "string", "enum": ["none", "demand_group", "abc", "location"]}}}},
    {"name": "value_impact",
     "description": "Dollar value of the software: one-time cash released, carrying-cost savings, recovered margin, "
                    "annual value, ROI and payback, for one way of taking the gain.",
     "input_schema": {"type": "object", "properties": {
         "mode": {"type": "string", "enum": ["keep_fill_rate", "keep_stock", "sliders"]}}}},
    {"name": "run_scenarios",
     "description": "Stress-test four stocking strategies (ERP min/max, Recommended plan, Recommended + buffer, "
                    "Re-plan on signals) in one or more futures. With no futures given, uses the saved/preset set. "
                    "Each future may change demand %, volatility, one category, lead times for the largest suppliers, "
                    "costs and prices. Returns the ranking (expected cost, worst-case regret, worst fill rate) and "
                    "per-future results.",
     "input_schema": {"type": "object", "properties": {
         "futures": {"type": "array", "items": {"type": "object", "properties": {
             "name": {"type": "string"}, "probability_pct": {"type": "number"},
             "demand_change_pct": {"type": "number"}, "volatility_x": {"type": "number"},
             "category": {"type": "string"}, "category_change_pct": {"type": "number"},
             "lead_time_plus_weeks": {"type": "number"}, "suppliers_hit_pct": {"type": "number"},
             "lead_time_variability_x": {"type": "number"}, "cost_change_pct": {"type": "number"},
             "price_change_pct": {"type": "number"}}, "required": ["name"]}},
         "include_base_case": {"type": "boolean", "description": "Add an unchanged base case for comparison. Default true."}}}},
    {"name": "financial_pnl",
     "description": "Customer P&L of using the software: year-1 and run-rate net impact on operating income, cash "
                    "released, NPV, payback month, benefit/cost, and the annual P&L for one year. Any assumption can "
                    "be overridden.",
     "input_schema": {"type": "object", "properties": {
         "mode": {"type": "string", "enum": ["keep_fill_rate", "keep_stock", "sliders"]},
         "year": {"type": "integer", "minimum": 1, "maximum": 5},
         "overrides": {"type": "object", "description": "Any of: months, go_live, ramp, cash_months, fee, "
                       "onboarding, internal_once, internal_yr, expedite, expedite_cut, writeoff, writeoff_cut, "
                       "planners, hours_saved, rate, opex_pct, discount (shares as decimals, e.g. 0.3)."}}}},
    {"name": "suggested_orders",
     "description": "Purchase orders the recommended policy would place today (parts at or below reorder point), "
                    "with totals by supplier and flags for orders below the supplier minimum.",
     "input_schema": {"type": "object", "properties": {
         "location_id": {"type": "string"}, "supplier_id": {"type": "string"},
         "abc": {"type": "string", "enum": ["A", "B", "C"]},
         "limit": {"type": "integer", "minimum": 1, "maximum": 25}}}},
    {"name": "draft_purchase_orders",
     "description": "DRAFT purchase orders for the user to confirm. Nothing is written until the user clicks Confirm. "
                    "Either pass explicit lines, or pass filters to draft all suggested orders that match.",
     "input_schema": {"type": "object", "properties": {
         "lines": {"type": "array", "items": {"type": "object", "properties": {
             "item_id": {"type": "string"}, "location_id": {"type": "string"}, "qty": {"type": "number"}},
             "required": ["item_id", "location_id", "qty"]}},
         "location_id": {"type": "string"}, "supplier_id": {"type": "string"},
         "abc": {"type": "string", "enum": ["A", "B", "C"]}}}},
    {"name": "draft_min_max_changes",
     "description": "DRAFT min/max changes in the ERP for the user to confirm. With no new values given, uses the "
                    "recommended reorder point (min) and order-up-to (max). Nothing is written until Confirm.",
     "input_schema": {"type": "object", "properties": {
         "changes": {"type": "array", "items": {"type": "object", "properties": {
             "item_id": {"type": "string"}, "location_id": {"type": "string"},
             "new_min": {"type": "number"}, "new_max": {"type": "number"}},
             "required": ["item_id", "location_id"]}}}, "required": ["changes"]}},
]


def to_jsonable(x: Any, rows: int = MAX_ROWS) -> Any:
    """Make tool results compact and JSON-safe: round floats, cap table rows."""
    if isinstance(x, pd.DataFrame):
        out = x.head(rows).replace({np.nan: None})
        recs = json.loads(out.to_json(orient="records", date_format="iso", double_precision=4))
        return {"rows": recs, "total_rows": int(len(x))} if len(x) > rows else recs
    if isinstance(x, pd.Series):
        return to_jsonable(x.to_dict(), rows)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, rows) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, rows) for v in x[:200]]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


def run(client, model: str, system: str, history: list[dict], handlers: dict[str, Callable[..., Any]],
        on_step: Callable[[str, dict, Any], None] | None = None, max_steps: int = MAX_STEPS):
    """Run one user turn. `history` holds prior messages and ends with the new user message.

    Returns (answer text, updated history, list of tool calls made).
    """
    msgs = list(history)
    calls = []
    for _ in range(max_steps):
        resp = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system, tools=TOOLS, messages=msgs)
        content = [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in resp.content]
        msgs.append({"role": "assistant", "content": content})
        uses = [b for b in content if b.get("type") == "tool_use"]
        if resp.stop_reason != "tool_use" or not uses:
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            return text or "(No answer.)", msgs, calls
        results = []
        for u in uses:
            fn = handlers.get(u["name"])
            try:
                out = fn(**(u.get("input") or {})) if fn else {"error": f"Unknown tool {u['name']}"}
                err = False
            except Exception as e:  # noqa: BLE001 - shown to the model so it can recover
                out, err = {"error": f"{type(e).__name__}: {e}"}, True
            out = to_jsonable(out)
            calls.append({"tool": u["name"], "input": u.get("input") or {}, "output": out})
            if on_step:
                on_step(u["name"], u.get("input") or {}, out)
            results.append({"type": "tool_result", "tool_use_id": u["id"], "is_error": err,
                            "content": json.dumps(out, default=str)[:60_000]})
        msgs.append({"role": "user", "content": results})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "I stopped after several steps without "
                                                   "finishing. Try a narrower question."}]})
    return msgs[-1]["content"][0]["text"], msgs, calls


def trim_history(msgs: list[dict], keep_turns: int = 6) -> list[dict]:
    """Keep the last few user questions (with their tool exchanges) to bound cost."""
    starts = [i for i, m in enumerate(msgs) if m["role"] == "user" and isinstance(m["content"], str)]
    return msgs[starts[-keep_turns]:] if len(starts) > keep_turns else msgs
