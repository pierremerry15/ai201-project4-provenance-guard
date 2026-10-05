"""
Automated checks for Provenance Guard. The LLM signal is mocked, so no API key or network
is needed:   python -m pytest -q   (or)   python tests/test_app.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key")

_tmp = tempfile.mkdtemp()
os.environ["AUDIT_LOG_FILE"] = os.path.join(_tmp, "audit_log.jsonl")
os.environ["DB_FILE"] = os.path.join(_tmp, "provenance.db")

import app as app_module  # noqa: E402
import scoring  # noqa: E402
import signals  # noqa: E402
from tests.samples import CLEARLY_AI, CLEARLY_HUMAN, EDITED_AI, FORMAL_HUMAN  # noqa: E402

client = app_module.app.test_client()


def mock_llm(score):
    def _fake(text):
        if score is None:
            return {"score": None, "status": "unavailable", "indicators": [], "reasoning": "",
                    "error": "APIConnectionError"}
        return {"score": score, "status": "ok", "indicators": ["mock"], "reasoning": "mock",
                "error": None}
    app_module.llm_signal = _fake


def submit(text, creator="tester"):
    app_module.limiter.reset()
    return client.post("/submit", json={"text": text, "creator_id": creator})


# ---------------------------------------------------------------- signals

def test_stylometry_separates_clear_cases():
    ai = signals.stylometric_signal(CLEARLY_AI)["score"]
    human = signals.stylometric_signal(CLEARLY_HUMAN)["score"]
    assert ai > 0.75 and human < 0.2, (ai, human)
    for text in (FORMAL_HUMAN, EDITED_AI):
        s = signals.stylometric_signal(text)["score"]
        assert 0.3 < s < 0.7, s


def test_llm_parser_handles_variants():
    assert signals._parse_llm_output('{"ai_likelihood": 0.9}')[0] == 0.9
    assert signals._parse_llm_output('```json\n{"ai_likelihood": 1.7}\n```')[0] == 1.0
    assert signals._parse_llm_output('Sure: {"reasoning":"x","ai_likelihood":"0.2"}')[0] == 0.2
    for bad in ("no json here", '{"score": 0.5}', '{"ai_likelihood": "high"}'):
        try:
            signals._parse_llm_output(bad)
            assert False, bad
        except (ValueError, KeyError):
            pass


# ---------------------------------------------------------------- scoring

def test_all_three_labels_reachable():
    mock_llm(0.92)
    r = submit(CLEARLY_AI).get_json()
    assert r["attribution"] == "likely_ai" and r["label"] == scoring.LABELS["likely_ai"], r
    mock_llm(0.05)
    r = submit(CLEARLY_HUMAN).get_json()
    assert r["attribution"] == "likely_human" and r["label"] == scoring.LABELS["likely_human"], r
    mock_llm(0.55)
    r = submit(FORMAL_HUMAN).get_json()
    assert r["attribution"] == "uncertain" and r["label"] == scoring.LABELS["uncertain"], r


def test_confidence_is_meaningfully_different():
    mock_llm(0.95)
    high = submit(CLEARLY_AI).get_json()
    mock_llm(0.5)
    low = submit(EDITED_AI).get_json()
    assert high["confidence"] - low["confidence"] > 0.25, (high["confidence"], low["confidence"])
    assert high["label"] != low["label"]


def test_ai_label_needs_both_signals():
    # Property: whenever the result is likely_ai, BOTH signals independently lean AI.
    for llm in [i / 100 for i in range(101)]:
        for st in [i / 100 for i in range(101)]:
            r = scoring.combine(llm, st, 200)
            if r["attribution"] == "likely_ai":
                assert llm >= 0.65 and st >= 0.55, (llm, st, r)
    # A very confident LLM alone can't produce an AI label when stylometry disagrees.
    assert scoring.combine(1.0, 0.50, 200)["attribution"] == "uncertain"
    # The explicit gate is a backstop if the weights are ever changed so the blend alone
    # would cross 0.80: it still blocks the label and explains why.
    orig = (scoring.LLM_WEIGHT, scoring.STYLO_WEIGHT)
    try:
        scoring.LLM_WEIGHT, scoring.STYLO_WEIGHT = 1.0, 0.0
        r = scoring.combine(1.0, 0.70, 200)  # blend ~0.85, but disagreement is only 0.30
        r2 = scoring.combine(1.0, 0.50, 200)  # blend ~0.88, stylometry below its 0.55 bar
        assert r2["attribution"] == "uncertain" and "stylometry alone" in r2["gate_note"], r2
        scoring.LLM_WEIGHT, scoring.STYLO_WEIGHT = 0.0, 1.0
        r3 = scoring.combine(0.64, 1.0, 200)  # blend ~0.91, LLM below its own 0.65 bar
        assert r3["attribution"] == "uncertain" and "LLM signal alone" in r3["gate_note"], r3
        assert r["attribution"] == "likely_ai"
    finally:
        scoring.LLM_WEIGHT, scoring.STYLO_WEIGHT = orig


def test_llm_unavailable_never_labels_ai():
    mock_llm(None)
    r = submit(CLEARLY_AI).get_json()
    assert r["attribution"] != "likely_ai", r
    assert r["signals_used"] == ["stylometry"] and "unavailable" in r["gate_note"]
    assert scoring.combine(None, 1.0, 500)["attribution"] == "uncertain"


def test_short_text_gates():
    assert scoring.combine(0.99, 0.99, 30)["attribution"] == "uncertain"   # < 40 words
    assert scoring.combine(0.01, 0.01, 20)["attribution"] == "uncertain"   # < 25 words
    assert scoring.combine(0.01, 0.01, 30)["attribution"] == "likely_human"


def test_score_and_label_never_contradict():
    for llm in [None] + [i / 20 for i in range(21)]:
        for st in [i / 10 for i in range(11)]:
            for words in (10, 30, 45, 200):
                r = scoring.combine(llm, st, words)
                s = r["ai_score"]
                expected = ("likely_ai" if s >= 0.80 else "likely_human" if s <= 0.35
                            else "uncertain")
                assert r["attribution"] == expected, (llm, st, words, r)
                assert abs(r["confidence"] - max(s, 1 - s)) < 1e-9


# ---------------------------------------------------------------- validation

def test_submit_validation():
    mock_llm(0.5)
    app_module.limiter.reset()
    assert client.post("/submit", data="nope").status_code == 400
    assert client.post("/submit", json={"creator_id": "a"}).status_code == 400
    assert client.post("/submit", json={"text": CLEARLY_AI}).status_code == 400
    assert client.post("/submit", json={"text": "too short", "creator_id": "a"}).status_code == 400
    assert client.post("/submit", json={"text": "x " * 6000, "creator_id": "a"}).status_code == 400


# ---------------------------------------------------------------- appeals + log

def test_appeal_flow_and_audit_log():
    mock_llm(0.92)
    sub = submit(CLEARLY_AI, creator="writer-7").get_json()
    cid = sub["content_id"]

    assert client.post("/appeal", json={"content_id": cid}).status_code == 400
    assert client.post("/appeal", json={"content_id": "nope",
                                        "creator_reasoning": "x"}).status_code == 404
    assert client.post("/appeal", json={"content_id": cid, "creator_reasoning": "mine",
                                        "creator_id": "someone-else"}).status_code == 403

    r = client.post("/appeal", json={"content_id": cid,
                                     "creator_reasoning": "I wrote this myself."})
    assert r.status_code == 200 and r.get_json()["status"] == "under_review"
    assert client.post("/appeal", json={"content_id": cid,
                                        "creator_reasoning": "again"}).status_code == 409

    assert client.get(f"/content/{cid}").get_json()["status"] == "under_review"
    queue = client.get("/appeals").get_json()["queue"]
    assert any(q["content_id"] == cid and q["creator_reasoning"] == "I wrote this myself."
               for q in queue)

    entries = client.get("/log?limit=500").get_json()["entries"]
    sub_entry = next(e for e in entries if e["event"] == "submission" and e["content_id"] == cid)
    app_entry = next(e for e in entries if e["event"] == "appeal" and e["content_id"] == cid)
    for k in ("timestamp", "attribution", "confidence", "llm_score", "stylometric_score",
              "status", "appeal_filed"):
        assert k in sub_entry, k
    assert app_entry["status"] == "under_review" and app_entry["appeal_reasoning"]
    assert app_entry["original_attribution"] == "likely_ai"

    with open(os.environ["AUDIT_LOG_FILE"]) as f:
        for line in f:
            json.loads(line)  # every line is one complete JSON object


# ---------------------------------------------------------------- rate limiting

def test_rate_limit_10_per_minute():
    mock_llm(0.5)
    app_module.limiter.reset()
    codes = [client.post("/submit", json={"text": EDITED_AI, "creator_id": "rate"}).status_code
             for _ in range(12)]
    assert codes == [200] * 10 + [429] * 2, codes
    r = client.post("/submit", json={"text": EDITED_AI, "creator_id": "rate"})
    assert r.get_json()["error"] == "rate_limited"
    app_module.limiter.reset()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
