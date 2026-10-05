"""
Confidence scoring and transparency labels (planning.md §3 "How the signals combine", §4, §5).
"""
from config import (
    LLM_WEIGHT, STYLO_WEIGHT, SINGLE_SIGNAL_SHRINK, SHORT_TEXT_WORDS, SHORT_TEXT_SHRINK,
    AI_THRESHOLD, HUMAN_THRESHOLD, AI_GATE_LLM_MIN, AI_GATE_STYLO_MIN, AI_GATE_MIN_WORDS,
    HUMAN_GATE_MIN_WORDS,
)

LABELS = {
    "likely_ai": (
        "🤖 Likely AI-generated. Our automated checks found strong signs that this text was "
        "written with AI. Automated detection can be wrong, and the creator can appeal this label."
    ),
    "likely_human": (
        "✍️ Likely human-written. Our automated checks found writing patterns typical of a person "
        "rather than AI. This is an automated assessment, not a guarantee."
    ),
    "uncertain": (
        "❔ Origin unclear. Our automated checks couldn't confidently tell whether this was written "
        "by a person or with AI. Short, formal, or edited writing often gets this result, so please "
        "don't treat it as a judgment either way."
    ),
}

LABEL_VARIANT_NAMES = {
    "likely_ai": "high_confidence_ai",
    "likely_human": "high_confidence_human",
    "uncertain": "uncertain",
}


def _shrink(score: float, factor: float) -> float:
    """Pull a score toward 0.5 (no information) by `factor` (1.0 = unchanged)."""
    return 0.5 + (score - 0.5) * factor


def combine(llm_score, stylo_score: float, words: int) -> dict:
    """
    Combine the two signal scores into one calibrated ai_score, then apply gates.

    Returns {"ai_score", "attribution", "confidence", "signals_used", "gate_note"}.
    ai_score: 0.0 = confidently human, 0.5 = coin flip, 1.0 = confidently AI.
    confidence: how sure we are of the returned attribution = max(ai_score, 1 - ai_score).
    """
    notes = []

    if llm_score is not None:
        raw = LLM_WEIGHT * llm_score + STYLO_WEIGHT * stylo_score
        disagreement = abs(llm_score - stylo_score)
        score = _shrink(raw, 1 - 0.5 * disagreement)
        signals_used = ["llm_judge", "stylometry"]
    else:
        score = _shrink(stylo_score, SINGLE_SIGNAL_SHRINK)
        signals_used = ["stylometry"]
        notes.append("LLM signal unavailable; scored on stylometry alone")

    if words < SHORT_TEXT_WORDS:
        score = _shrink(score, SHORT_TEXT_SHRINK)

    # --- Gates (asymmetric: an AI label needs more evidence than a human label) ---
    if score >= AI_THRESHOLD:
        reasons = []
        if llm_score is None:
            reasons.append("LLM signal unavailable")
        else:
            if llm_score < AI_GATE_LLM_MIN:
                reasons.append(f"LLM signal alone ({llm_score:.2f}) is below {AI_GATE_LLM_MIN}")
            if stylo_score < AI_GATE_STYLO_MIN:
                reasons.append(f"stylometry alone ({stylo_score:.2f}) is below {AI_GATE_STYLO_MIN}")
        if words < AI_GATE_MIN_WORDS:
            reasons.append(f"text is shorter than {AI_GATE_MIN_WORDS} words")
        if reasons:
            score = AI_THRESHOLD - 0.01
            notes.append("Not labeled AI: " + "; ".join(reasons))

    if score <= HUMAN_THRESHOLD and words < HUMAN_GATE_MIN_WORDS:
        score = HUMAN_THRESHOLD + 0.01
        notes.append(f"Not labeled human: text is shorter than {HUMAN_GATE_MIN_WORDS} words")

    score = round(min(1.0, max(0.0, score)), 3)
    if score >= AI_THRESHOLD:
        attribution = "likely_ai"
    elif score <= HUMAN_THRESHOLD:
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    return {
        "ai_score": score,
        "attribution": attribution,
        "confidence": round(max(score, 1 - score), 3),
        "signals_used": signals_used,
        "gate_note": "; ".join(notes) or None,
    }


def label_for(attribution: str) -> str:
    return LABELS.get(attribution, LABELS["uncertain"])
