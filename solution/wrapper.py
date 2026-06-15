"""Mitigation + observability wrapper.
call_next(question, config) -> result is the ONLY way to reach the agent.
context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}
"""
from __future__ import annotations
import time
import re
import json
import os
import sys

# Safe telemetry imports
try:
    from telemetry.logger import logger, new_correlation_id, set_correlation_id
    from telemetry.cost import cost_from_usage
    _HAS_TELEMETRY = True
except Exception:
    _HAS_TELEMETRY = False

# ── PII patterns ──
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b(?:\+84|0)\d{9}\b")
_CCCD_RE  = re.compile(r"\b\d{12}\b")

# ── Injection note patterns ──
_NOTE_RE = re.compile(
    r"(?:GHI\s*CH[UÚ]|note|ghi\s*ch[uú]|ch[uú]\s*[yý])\s*[:：].*",
    re.IGNORECASE | re.DOTALL,
)


def _redact_answer(answer):
    if not answer:
        return answer
    answer = _EMAIL_RE.sub("[REDACTED]", answer)
    answer = _PHONE_RE.sub("[REDACTED]", answer)
    answer = _CCCD_RE.sub("[REDACTED]", answer)
    return answer


def _sanitize_question(question):
    return _NOTE_RE.sub("", question).strip()


def _log(event_type, data):
    if _HAS_TELEMETRY:
        try:
            logger.log_event(event_type, data)
        except Exception:
            pass


def mitigate(call_next, question, config, context):
    qid = context.get("qid", "")
    session_id = context.get("session_id", "")
    turn_index = context.get("turn_index", 0)

    if _HAS_TELEMETRY:
        try:
            cid = new_correlation_id()
            set_correlation_id(cid)
        except Exception:
            pass

    # ── 1. Cache ──
    cache = context.get("cache", {})
    cache_lock = context.get("cache_lock")
    cache_key = question.strip().lower()

    try:
        if cache_lock:
            with cache_lock:
                cached = cache.get(cache_key)
        else:
            cached = cache.get(cache_key)
        if cached is not None:
            _log("CACHE_HIT", {"qid": qid, "session_id": session_id})
            return cached
    except Exception:
        pass

    # ── 2. Sanitize question ──
    clean_q = _sanitize_question(question)

    # ── 3. Config ──
    conf = dict(config)

    # ── 4. Call agent with retry ──
    max_retries = 2
    result = None
    last_error = None
    attempts_used = 0
    t0 = time.time()

    for attempt in range(max_retries):
        attempts_used = attempt
        try:
            result = call_next(clean_q, conf)
            status = result.get("status", "")
            if status == "ok":
                break
            # On loop/max_steps, try once more
            if status in ("loop", "max_steps") and attempt < max_retries - 1:
                last_error = status
                continue
            break
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(0.3)
            continue

    wall_ms = int((time.time() - t0) * 1000)

    if result is None:
        result = {
            "answer": None,
            "status": "wrapper_error",
            "steps": 0,
            "trace": [],
            "meta": {},
        }

    # ── 5. Redact PII ──
    if result.get("answer"):
        result["answer"] = _redact_answer(result["answer"])

    # ── 6. Cache ok results ──
    if result.get("status") == "ok":
        try:
            if cache_lock:
                with cache_lock:
                    cache[cache_key] = result
            else:
                cache[cache_key] = result
        except Exception:
            pass

    # ── 7. Observability ──
    meta = result.get("meta", {})
    usage = meta.get("usage", {})
    model = meta.get("model", conf.get("model", "unknown"))
    cost = 0.0
    if _HAS_TELEMETRY:
        try:
            cost = cost_from_usage(model, usage)
        except Exception:
            pass

    _log("CALL", {
        "qid": qid, "session_id": session_id, "turn_index": turn_index,
        "wall_ms": wall_ms, "latency_ms": meta.get("latency_ms"),
        "status": result.get("status"), "steps": result.get("steps"),
        "tools_used": meta.get("tools_used", []),
        "tool_count": len(meta.get("tools_used", [])),
        "usage": usage, "cost_usd": cost, "model": model,
        "retries": attempts_used, "error": last_error,
    })

    return result
