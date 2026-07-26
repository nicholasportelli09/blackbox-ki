"""
Portfolio narrator — turns a user's REAL portfolio data (NAV, day-return,
per-strategy notes, tilts — the same figures already shown in the UI, built
by app_user._user_payload) into a plain-English write-up, plus a Q&A helper
that answers questions using that same data as its only source of truth.

AI never invents a figure — every number it writes about is passed to it
verbatim in the context below. Fails open: returns None when AI is
unavailable, so the caller can hide the feature rather than show an error.
"""
import json

import ai_client

SYSTEM = (
    "You are a plain-English explainer inside a paper-trading investing app called "
    "Blackbox. You are given a user's REAL portfolio data as JSON — NAV, returns, "
    "strategy weights and notes already computed by validated, backtested rules. "
    "Rules you must follow: (1) NEVER state a number that is not present in the "
    "data given to you. (2) NEVER give personalized financial advice or tell the "
    "user what to buy, sell, or do with their money — you explain what already "
    "happened, you do not recommend what happens next. (3) If a question can't be "
    "answered from the data given, say so plainly instead of guessing. (4) This is "
    "practice money, not a real account — stay factual and measured, never hyped."
)


def _context(payload):
    slim = {
        "name": payload.get("name"), "nav": payload.get("nav"),
        "day_return_pct": payload.get("day_ret"), "since_start_pct": payload.get("since_start"),
        "created": payload.get("created"),
        "strategies": [
            {"id": s.get("id"), "weight_pct": s.get("weight"),
             "day_return_pct": round((s.get("day_ret") or 0) * 100, 3),
             "note": s.get("note"), "tilt": s.get("tilt"), "tilt_why": s.get("tilt_why")}
            for s in payload.get("strategies", [])
        ],
    }
    return json.dumps(slim)


def narrate(payload):
    """Short plain-English summary of this portfolio's current state. None if
    AI is unavailable."""
    if not ai_client.available():
        return None
    prompt = (f"Portfolio data:\n{_context(payload)}\n\n"
              "Write a short (3-4 sentence) plain-English summary of how this "
              "portfolio is doing right now and why, based only on the data above.")
    return ai_client.ask(prompt, system=SYSTEM, max_tokens=300)


def answer(payload, question):
    """Answer a free-text question about this portfolio, grounded only in the
    data above. None if AI is unavailable or the question is empty."""
    if not ai_client.available() or not (question or "").strip():
        return None
    question = question.strip()[:400]
    prompt = (f'Portfolio data:\n{_context(payload)}\n\nUser question: "{question}"\n\n'
              "Answer in 2-4 sentences, using only the data above.")
    return ai_client.ask(prompt, system=SYSTEM, max_tokens=350)
