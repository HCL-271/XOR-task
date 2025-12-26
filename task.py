
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import random
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from anthropic.types import ToolUnionParam


_CASE_FILES = ["case1_basic.json", "case2_vars.json"]
_CASE_DIR = Path(__file__).parent


def _load_cases() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for fn in _CASE_FILES:
        p = _CASE_DIR / fn
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "case_id" not in data:
            raise ValueError(f"Invalid case file {fn}")
        cases[str(data["case_id"])] = data
    print(cases, "cases 26")
    return cases


CASES: dict[str, dict[str, Any]] = _load_cases()

# Bias towards the harder variable-driven case to keep pass rate < 70%.
_CASE_WEIGHTS: dict[str, float] = {
    "case1_basic": 0.4,
    "case2_vars": 0.6,
}


def _pick_case_id() -> str:
    ids = list(CASES.keys())
    weights = [_CASE_WEIGHTS.get(cid, 1.0) for cid in ids]
    res = random.choices(ids, weights=weights, k=1)[0]
    print(res, "random 44")
    return res

_SECRET = b"m,65nm,54,mn;546"


def _sig(case_id: str) -> str:
    mac = hmac.new(_SECRET, case_id.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def make_token(case_id: str) -> str:
    return f"{case_id}.{_sig(case_id)}"


def verify_token(token: str, case_id: str) -> bool:
    if not isinstance(token, str) or "." not in token:
        return False
    cid, sig = token.split(".", 1)
    print(cid, sig, "cid, sig 64")
    return cid == case_id and hmac.compare_digest(sig, _sig(case_id))

class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def get_case_tool() -> dict[str, Any]:
    # I have wto cases, so I need to choose between them
    cid = _pick_case_id()
    case = CASES[cid]
    return {
        "case_id": case["case_id"],
        "token": make_token(case["case_id"]),
        "description": case.get("description", ""),
        "as_of_date": case["as_of_date"],
        "as_of_ts": case["as_of_ts"],
        "procedure_sql": case["procedure_sql"],
        "inputs": case["inputs"],
        "required_target_cols": case["required_target_cols"],
        "allowed_transforms": case["allowed_transforms"],
        "rules": [
            "Return ONLY a submit_answer tool call. No normal text.",
            "Use fully-qualified src cols: dds.dim_agree.<column>",
            "For transform=direct_copy: src_cols must have exactly 1 element",
            "For transform=system_etl_updated_dttm: src_cols must be []",
            "For calc_is_active: include params.active_states extracted from SQL",
            "For calc_open_rub_amt: include params.base_currency, params.rates, params.default_rate extracted from SQL",
        ],
    }


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    return {"answer": answer, "submitted": True}


# some giant structure
REQUIRED_COLS = [
    "agree_sk",
    "agree_open_dt",
    "agree_close_dt",
    "agree_state",
    "is_active",
    "agree_age_days",
    "agree_duration_day_cnt",
    "open_rub_amt",
    "etl_updated_dttm",
]

ALLOWED_TRANSFORMS = [
    "direct_copy",
    "calc_is_active",
    "calc_agree_age_days",
    "calc_agree_duration_day_cnt",
    "calc_open_rub_amt",
    "system_etl_updated_dttm",
]

TOOLS: list[ToolUnionParam] = [
    {
        "name": "get_case",
        "description": "Get one ETL case (SQL + input rows).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_answer",
        "description": "Submit the final answer. OUTPUT ONLY THIS TOOL CALL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "token": {"type": "string"},
                        "output_keys": {"type": "array", "items": {"type": "integer"}},
                        "mappings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tgt_col": {"type": "string", "enum": REQUIRED_COLS},
                                    "src_cols": {"type": "array", "items": {"type": "string"}},
                                    "transform": {"type": "string", "enum": ALLOWED_TRANSFORMS},
                                    "params": {"type": "object"},
                                },
                                "required": ["tgt_col", "src_cols", "transform"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["case_id", "token", "output_keys", "mappings"],
                    "additionalProperties": False,
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "get_case": get_case_tool,
    "submit_answer": submit_answer_tool,
}



PROMPT = """
Call get_case().
Then immediately call submit_answer with JSON under key "answer" (no normal text).

Your answer JSON must contain:
- case_id, token (copy from get_case)
- output_keys: list[int] of agree_sk that survive the WHERE clause
- mappings: one entry for every required_target_cols

Params requirement:
- If transform == calc_is_active: mapping MUST include params.active_states extracted from SQL
- If transform == calc_open_rub_amt: mapping MUST include params.base_currency ('RUB'), params.rates (USD/EUR), params.default_rate extracted from SQL

Hard rules:
- Output ONLY a submit_answer tool call.
- Do NOT include explanations or SQL.
""".strip()



def _parse_date(s: str | None) -> date | None:
    if s is None:
        return None
    return date.fromisoformat(s)


def _parse_ts(s: str) -> datetime:
    # Accept "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(s)


def _get_row_by_key(rows: list[dict[str, Any]], key: str, val: Any) -> dict[str, Any] | None:
    for r in rows:
        if r.get(key) == val:
            return r
    return None


def _get_src_value(dim_agree_row: dict[str, Any] | None, src: str) -> Any:
    # src should be "dds.dim_agree.<col>"
    if dim_agree_row is None:
        return None
    if not isinstance(src, str) or src.count(".") < 2:
        return None
    table, col = src.rsplit(".", 1)
    if table != "dds.dim_agree":
        return None
    return dim_agree_row.get(col)


def _find_suffix(src_cols: list[str], suffix: str) -> str | None:
    # 7 usages to left in code
    for s in src_cols:
        if isinstance(s, str) and s.endswith("." + suffix):
            return s
    return None


def _required_sources_ok(transform: str, src_cols: list[str]) -> bool:
    # Minimal "must include" checks to prevent random mappings passing.
    if transform == "direct_copy":
        return len(src_cols) == 1 and src_cols[0].startswith("dds.dim_agree.")
    if transform == "system_etl_updated_dttm":
        return len(src_cols) == 0

    must: list[str] = []
    if transform == "calc_is_active":
        must = ["agree_state"]
    elif transform == "calc_agree_age_days":
        must = ["open_dt"]
    elif transform == "calc_agree_duration_day_cnt":
        must = ["open_dt", "close_dt"]
    elif transform == "calc_open_rub_amt":
        must = ["currency_name", "open_amt"]

    return all(_find_suffix(src_cols, m) is not None for m in must)


def _compute(
    transform: str,
    src_cols: list[str],
    params: dict[str, Any] | None,
    row: dict[str, Any],
    as_of_d: date,
    as_of_ts: str,
) -> Any | None:
    if transform == "direct_copy":
        if len(src_cols) != 1:
            return None
        return _get_src_value(row, src_cols[0])

    if transform == "system_etl_updated_dttm":
        if len(src_cols) != 0:
            return None
        print(as_of_ts, "as_of_ts 267")
        return as_of_ts

    if transform == "calc_is_active":
        if not isinstance(params, dict):
            return None
        active_states = params.get("active_states")
        if not isinstance(active_states, list) or not all(isinstance(x, str) for x in active_states):
            return None
        s = _find_suffix(src_cols, "agree_state")
        st = _get_src_value(row, s) if s else None
        print( st, "st 277")
        return True if st in set(active_states) else False

    if transform == "calc_agree_age_days":
        s = _find_suffix(src_cols, "open_dt")
        od = _parse_date(_get_src_value(row, s)) if s else None
        if od is None:
            return None
        return (as_of_d - od).days

    if transform == "calc_agree_duration_day_cnt":
        s_open = _find_suffix(src_cols, "open_dt")
        s_close = _find_suffix(src_cols, "close_dt")
        od = _parse_date(_get_src_value(row, s_open)) if s_open else None
        if od is None:
            return None
        cd = _parse_date(_get_src_value(row, s_close)) if s_close else None
        if cd is not None:
            return (cd - od).days
        return (as_of_d - od).days

    if transform == "calc_open_rub_amt":
        if not isinstance(params, dict):
            return None
        base = params.get("base_currency")
        if base != "RUB":
            return None
        rates = params.get("rates")
        default_rate = params.get("default_rate")
        if not isinstance(rates, dict) or not all(isinstance(k, str) for k in rates.keys()):
            return None
        if not isinstance(default_rate, int):
            return None
        # require at least USD/EUR keys in rates
        if "USD" not in rates or "EUR" not in rates:
            return None
        if not isinstance(rates["USD"], int) or not isinstance(rates["EUR"], int):
            return None

        s_cur = _find_suffix(src_cols, "currency_name")
        s_amt = _find_suffix(src_cols, "open_amt")
        cur = _get_src_value(row, s_cur) if s_cur else None
        amt = _get_src_value(row, s_amt) if s_amt else None
        print(s_cur, s_amt, cur, amt, "(s_cur, s_amt, cur, amt, 321")
        if cur is None or amt is None:
            return None
        if cur == base:
            return amt
        mult = rates.get(cur, default_rate)
        if not isinstance(mult, int):
            return None
        return amt * mult

    return None


def _extract_params_for_transform(case: dict[str, Any], transform: str) -> dict[str, Any] | None:
    exp = case.get("expected_params")
    if not isinstance(exp, dict):
        return None
    p = exp.get(transform)
    return p if isinstance(p, dict) else None



def grading_func(result: Any) -> bool:
    ans: Any = result
    if ans is None:
        return False
    try:
        if isinstance(ans, str):
            ans = json.loads(ans)
    except Exception:
        return False

    if not isinstance(ans, dict):
        return False

    case_id = ans.get("case_id")
    token = ans.get("token")
    if not isinstance(case_id, str) or case_id not in CASES:
        return False
    case = CASES[case_id]

    if not isinstance(token, str) or not verify_token(token, case_id):
        return False

    # Check output_keys (filters)
    output_keys = ans.get("output_keys")
    print(output_keys)
    if not isinstance(output_keys, list) or not all(isinstance(x, int) for x in output_keys):
        return False

    expected_keys = sorted({row["agree_sk"] for row in case["expected_output"]})
    print(expected_keys)
    if sorted(set(output_keys)) != expected_keys:
        return False

    # Parse mappings
    mappings = ans.get("mappings")
    print(mappings)
    if not isinstance(mappings, list):
        return False

    allowed_transforms = set(case["allowed_transforms"])
    required_cols = list(case["required_target_cols"])

    by_tgt: dict[str, dict[str, Any]] = {}
    for m in mappings:
        if not isinstance(m, dict):
            continue
        t = m.get("tgt_col")
        if not isinstance(t, str):
            continue
        if t in by_tgt:
            return False  # no duplicates
        by_tgt[t] = m

    print(required_cols)
    for col in required_cols:
        if col not in by_tgt:
            return False

    # Validate mappings
    for col in required_cols:
        m = by_tgt[col]
        transform = m.get("transform")
        src_cols = m.get("src_cols")
        params = m.get("params") if "params" in m else None

        if transform not in allowed_transforms:
            return False
        if not isinstance(src_cols, list) or not all(isinstance(s, str) for s in src_cols):
            return False
        if not _required_sources_ok(transform, src_cols):
            return False


        if transform in ("calc_is_active", "calc_open_rub_amt"):
            if not isinstance(params, dict):
                return False
            exp_params = _extract_params_for_transform(case, transform)
            if not isinstance(exp_params, dict):
                return False


            if transform == "calc_is_active":
                got_states = params.get("active_states")
                exp_states = exp_params.get("active_states")
                if (
                    not isinstance(got_states, list)
                    or not all(isinstance(x, str) for x in got_states)
                    or not isinstance(exp_states, list)
                ):
                    return False
                if set(got_states) != set(exp_states):
                    return False

            if transform == "calc_open_rub_amt":
                # Require exact expected ints
                if params.get("base_currency") != exp_params.get("base_currency"):
                    return False
                if params.get("default_rate") != exp_params.get("default_rate"):
                    return False
                got_rates = params.get("rates")
                exp_rates = exp_params.get("rates")
                if not isinstance(got_rates, dict) or not isinstance(exp_rates, dict):
                    return False
                for k in ("USD", "EUR"):
                    if got_rates.get(k) != exp_rates.get(k):
                        return False

        else:
            # For other transforms, params is ignored if provided.
            pass

    as_of_d = _parse_date(case["as_of_date"])
    if as_of_d is None:
        return False
    as_of_ts = case["as_of_ts"]

    dim_agree = case["inputs"]["dds.dim_agree"]
    print(dim_agree)
    for exp_row in case["expected_output"]:
        agree_sk = exp_row["agree_sk"]
        src_row = _get_row_by_key(dim_agree, "agree_sk", agree_sk)
        if src_row is None:
            return False

        for col in required_cols:
            m = by_tgt[col]
            got = _compute(
                m["transform"],
                m["src_cols"],
                m.get("params"),
                src_row,
                as_of_d,
                as_of_ts,
            )
            exp = exp_row.get(col)
            print(exp)
            if got != exp:
                return False

    return True
