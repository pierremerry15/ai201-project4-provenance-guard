"""
Provenance Guard: AI-text attribution API.

POST /submit             classify a piece of text -> attribution, confidence, transparency label
POST /appeal             creator contests a classification -> status "under_review"
GET  /content/<id>       current state of one submission
GET  /appeals            human-review queue
GET  /log                most recent audit-log entries

Design spec: planning.md
"""
import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import config
import storage
from scoring import LABEL_VARIANT_NAMES, combine, label_for
from signals import llm_signal, stylometric_signal

app = Flask(__name__)
app.json.ensure_ascii = False  # keep the emoji in labels readable in responses

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

storage.init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(status: int, error: str, detail: str):
    return jsonify({"error": error, "detail": detail}), status


@app.errorhandler(429)
def rate_limited(e):
    return _error(429, "rate_limited", f"Rate limit exceeded: {e.description}. Try again later.")


@app.errorhandler(404)
def not_found(e):
    return _error(404, "not_found", "No such endpoint.")


@app.errorhandler(405)
def method_not_allowed(e):
    return _error(405, "method_not_allowed", "Wrong HTTP method for this endpoint.")


def analyze(text: str) -> dict:
    """Run both signals and the scorer. Pure pipeline: no storage, no logging."""
    llm = llm_signal(text)
    stylo = stylometric_signal(text)
    words = stylo["metrics"]["words"]
    result = combine(llm["score"], stylo["score"], words)
    result["label"] = label_for(result["attribution"])
    result["signals"] = {
        "llm_judge": {k: llm[k] for k in ("score", "status", "indicators", "reasoning")},
        "stylometry": {"score": stylo["score"], "components": stylo["components"],
                       "metrics": stylo["metrics"]},
    }
    result["llm_score"] = llm["score"]
    result["llm_status"] = llm["status"]
    result["stylometric_score"] = stylo["score"]
    result["word_count"] = words
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return jsonify({
        "service": "Provenance Guard",
        "endpoints": {
            "POST /submit": "{text, creator_id} -> attribution, confidence, label",
            "POST /appeal": "{content_id, creator_reasoning[, creator_id]} -> under_review",
            "GET /content/<content_id>": "current state of a submission",
            "GET /appeals": "human review queue",
            "GET /log?limit=N": "most recent audit-log entries",
        },
    })


@app.post("/submit")
@limiter.limit(config.SUBMIT_RATE_LIMIT)
def submit():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error(400, "invalid_json", "Send a JSON body with 'text' and 'creator_id'.")

    text = body.get("text")
    creator_id = body.get("creator_id")
    if not isinstance(text, str) or not text.strip():
        return _error(400, "missing_text", "'text' is required and must be a non-empty string.")
    if not isinstance(creator_id, str) or not creator_id.strip():
        return _error(400, "missing_creator_id", "'creator_id' is required.")
    text, creator_id = text.strip(), creator_id.strip()[:100]
    if len(text) < config.MIN_TEXT_CHARS:
        return _error(400, "text_too_short",
                      f"'text' must be at least {config.MIN_TEXT_CHARS} characters.")
    if len(text) > config.MAX_TEXT_CHARS:
        return _error(400, "text_too_long",
                      f"'text' must be at most {config.MAX_TEXT_CHARS} characters.")

    result = analyze(text)
    content_id = str(uuid.uuid4())
    timestamp = storage.utc_now()
    status = "classified"

    storage.save_content({
        "content_id": content_id, "creator_id": creator_id, "text": text,
        "attribution": result["attribution"], "confidence": result["confidence"],
        "ai_score": result["ai_score"], "llm_score": result["llm_score"],
        "stylometric_score": result["stylometric_score"], "signals": result["signals"],
        "gate_note": result["gate_note"], "label": result["label"], "status": status,
        "timestamp": timestamp,
    })

    storage.append_audit({
        "event": "submission",
        "timestamp": timestamp,
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "ai_score": result["ai_score"],
        "llm_score": result["llm_score"],
        "llm_status": result["llm_status"],
        "stylometric_score": result["stylometric_score"],
        "signals_used": result["signals_used"],
        "gate_note": result["gate_note"],
        "label_variant": LABEL_VARIANT_NAMES[result["attribution"]],
        "status": status,
        "appeal_filed": False,
        "word_count": result["word_count"],
        "text_preview": text[:200],
        "model": config.GROQ_MODEL,
    })

    print(f"[SUBMIT] {content_id[:8]} creator={creator_id} -> {result['attribution']} "
          f"ai_score={result['ai_score']} (llm={result['llm_score']}, "
          f"stylo={result['stylometric_score']})")

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "ai_score": result["ai_score"],
        "label": result["label"],
        "status": status,
        "signals": result["signals"],
        "signals_used": result["signals_used"],
        "gate_note": result["gate_note"],
        "timestamp": timestamp,
    })


@app.post("/appeal")
@limiter.limit(config.APPEAL_RATE_LIMIT)
def appeal():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error(400, "invalid_json",
                      "Send a JSON body with 'content_id' and 'creator_reasoning'.")

    content_id = body.get("content_id")
    reasoning = body.get("creator_reasoning")
    claimed_creator = body.get("creator_id")
    if not isinstance(content_id, str) or not content_id.strip():
        return _error(400, "missing_content_id", "'content_id' is required.")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return _error(400, "missing_creator_reasoning",
                      "'creator_reasoning' is required: explain why the label is wrong.")
    reasoning = reasoning.strip()
    if len(reasoning) > config.MAX_APPEAL_CHARS:
        return _error(400, "reasoning_too_long",
                      f"'creator_reasoning' must be at most {config.MAX_APPEAL_CHARS} characters.")

    record = storage.get_content(content_id.strip())
    if record is None:
        return _error(404, "content_not_found", f"No submission with content_id {content_id!r}.")
    if claimed_creator is not None and str(claimed_creator).strip() != record["creator_id"]:
        return _error(403, "not_creator", "Only the original creator can appeal this classification.")

    appealed_at = storage.utc_now()
    if not storage.mark_under_review(record["content_id"], reasoning, appealed_at):
        return _error(409, "already_under_review",
                      "An appeal for this content is already under review.")

    appeal_id = str(uuid.uuid4())
    storage.append_audit({
        "event": "appeal",
        "timestamp": appealed_at,
        "appeal_id": appeal_id,
        "content_id": record["content_id"],
        "creator_id": record["creator_id"],
        "status": "under_review",
        "appeal_filed": True,
        "appeal_reasoning": reasoning,
        "original_attribution": record["attribution"],
        "original_confidence": record["confidence"],
        "original_ai_score": record["ai_score"],
        "llm_score": record["llm_score"],
        "stylometric_score": record["stylometric_score"],
        "original_timestamp": record["submitted_at"],
    })

    print(f"[APPEAL] {record['content_id'][:8]} -> under_review "
          f"(was {record['attribution']}, ai_score={record['ai_score']})")

    return jsonify({
        "appeal_id": appeal_id,
        "content_id": record["content_id"],
        "status": "under_review",
        "original_attribution": record["attribution"],
        "original_confidence": record["confidence"],
        "message": "Appeal received. This content is now under review by a person. "
                   "The current label stays visible until a reviewer decides.",
        "timestamp": appealed_at,
    })


@app.get("/content/<content_id>")
def get_content(content_id):
    record = storage.get_content(content_id)
    if record is None:
        return _error(404, "content_not_found", f"No submission with content_id {content_id!r}.")
    return jsonify(record)


@app.get("/appeals")
def get_appeals():
    queue = []
    for r in storage.appeal_queue():
        queue.append({
            "content_id": r["content_id"],
            "creator_id": r["creator_id"],
            "submitted_at": r["submitted_at"],
            "appealed_at": r["appealed_at"],
            "attribution": r["attribution"],
            "label": r["label"],
            "ai_score": r["ai_score"],
            "confidence": r["confidence"],
            "llm_score": r["llm_score"],
            "stylometric_score": r["stylometric_score"],
            "llm_indicators": r["signals"]["llm_judge"].get("indicators", []),
            "stylometry_metrics": r["signals"]["stylometry"].get("metrics", {}),
            "gate_note": r["gate_note"],
            "text_excerpt": r["text"][:500],
            "creator_reasoning": r["appeal_reasoning"],
        })
    return jsonify({"count": len(queue), "queue": queue})


@app.get("/log")
def get_log():
    try:
        limit = max(1, min(500, int(request.args.get("limit", 50))))
    except ValueError:
        limit = 50
    return jsonify({"entries": storage.read_audit(limit)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
