"""
Free-text quiz nudge — the 6-question quiz builds a validated, deterministic
mix (api_recommend() in app_user.py). This module lets an optional free-text
answer nudge that mix afterward, WITHIN HARD BOUNDS:

  - AI may only apply a multiplier in [TILT_MIN, TILT_MAX] to a strategy
    ALREADY present in the base mix.
  - AI can never add a strategy the base formula excluded (e.g. crypto's
    eligibility gates still hold — there is nothing to multiply if the base
    weight is zero) and can never remove one outright.

This mirrors the existing sentiment-tilt pattern used elsewhere in the app
(a bounded multiplier on validated weights, never a free-form rewrite).
Fails open: returns None on any failure, caller keeps the base mix.
"""
import ai_client

TILT_MIN, TILT_MAX = 0.8, 1.2


def nudge(mix, parts, freetext):
    """mix: {strategy_id: weight_int}. parts: list of dicts with at least
    'id' and 'name' for strategies in the mix. freetext: the user's raw text.
    Returns (new_mix, reason) or None if no adjustment was made/possible."""
    if not ai_client.available() or not (freetext or "").strip():
        return None
    names = {p["id"]: p["name"] for p in parts}
    menu = "\n".join(f"- {sid} ({names.get(sid, sid)}): currently {w}% of the mix"
                      for sid, w in mix.items())
    prompt = (
        "A user already has a validated, backtested strategy mix built from a structured "
        "risk questionnaire — the numbers below are real and already correct. They also "
        "wrote a free-text note about their situation. You may propose a SMALL adjustment "
        f"(a multiplier between {TILT_MIN} and {TILT_MAX}) to any strategy ALREADY in the "
        "mix below, if their note reveals something the questionnaire didn't capture. Do "
        "NOT add a strategy that isn't already listed, and do not try to remove one "
        "entirely — if nothing in their note warrants a change, propose no tilts at all.\n\n"
        f"CURRENT MIX:\n{menu}\n\n"
        f'USER\'S FREE-TEXT NOTE: "{freetext.strip()[:600]}"\n\n'
        'Reply with ONLY this JSON, nothing else: {"tilts": {"<strategy id>": <multiplier '
        f'{TILT_MIN}-{TILT_MAX}>, ... only strategies you want to adjust, can be empty}}, '
        '"reason": "<one plain-English sentence on what in their note drove the change, '
        'or empty string if you proposed no tilts>"}'
    )
    result = ai_client.ask_json(prompt, max_tokens=400)
    if not result or not isinstance(result.get("tilts"), dict):
        return None
    tilts = {}
    for k, v in result["tilts"].items():
        if k not in mix:
            continue
        try:
            tilts[k] = max(TILT_MIN, min(TILT_MAX, float(v)))
        except (TypeError, ValueError):
            continue
    if not tilts:
        return None
    w = {k: v * tilts.get(k, 1.0) for k, v in mix.items()}
    tot = sum(w.values())
    if tot <= 0:
        return None
    fl = {k: int(v * 100 / tot) for k, v in w.items()}
    rem = 100 - sum(fl.values())
    for k in sorted(w, key=lambda k: -(w[k] * 100 / tot - fl[k]))[:rem]:
        fl[k] += 1
    new_mix = {k: v for k, v in sorted(fl.items(), key=lambda z: -z[1]) if v > 0}
    reason = str(result.get("reason", "")).strip()[:300]
    if not reason:
        return None
    return new_mix, reason
