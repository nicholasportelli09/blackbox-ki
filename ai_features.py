"""
AI content features — the briefing, tutor, glossary, strategy deep-dive and
comparison. All prose is generated at request time from REAL data (the live
catalog, the live sentiment scan, the historical-analog snapshot); AI writes
explanations and narrative, never performance numbers — every figure the user
sees still traces to catalog.json, per the project's first rule.

Everything fails open (returns None) so the UI degrades to "unavailable" rather
than erroring, and callers cache results to respect Groq's free-tier limits.
"""
import os, json

import ai_client

HERE = os.path.dirname(__file__)
CATALOG = os.path.join(HERE, "logs", "catalog.json")
SENTIMENT = os.path.join(HERE, "logs", "sentiment.json")
ANALOG = os.path.join(HERE, "logs", "historical_analog.json")

GUARD = ("You are the in-app analyst for Blackbox, a paper-trading platform of 8 "
         "validated, backtested systematic strategies. Rules: never invent a "
         "performance number — only use figures given to you. Never give personalized "
         "financial advice or say what to buy/sell. This is practice money, not real. "
         "Be clear, precise and educational; plain English, no hype.")


def _load(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def _strategies():
    return _load(CATALOG, {"strategies": []}).get("strategies", [])


def _strat(sid):
    return next((s for s in _strategies() if s["id"] == sid), None)


def _stat_line(s):
    st = s.get("stats", {})
    return (f"{s['name']} (risk: {s.get('risk','?')}): CAGR {st.get('cagr')}%, "
            f"Sharpe {st.get('sharpe')}, volatility {st.get('vol')}%, max drawdown "
            f"{st.get('max_dd')}%, beta vs S&P {st.get('beta')}. Method: {s.get('method','')}")


# ---------------------------------------------------------------- briefing
def briefing():
    """A plain-English market briefing synthesising the real sentiment scan and
    historical-analog snapshot. Returns dict or None."""
    if not ai_client.available():
        return None
    sent = _load(SENTIMENT, {})
    assets = sent.get("assets", {})
    if not assets:
        return None
    lines = [f"- {a}: score {v.get('score')} ({v.get('trend','')}), {v.get('explanation','')}"
             for a, v in assets.items()]
    analog = _load(ANALOG, {})
    an = ""
    if analog.get("event"):
        an = (f"\nClosest historical analog to today's news: {analog['event'].get('title')} "
              f"({analog['event'].get('date')}). {analog.get('reason','')}")
    prompt = (
        "Write today's market briefing for the user, in 3 short paragraphs. Use ONLY the "
        "AI sentiment reads below (scores run -100 bearish to +100 bullish) and the "
        "historical analog. Paragraph 1: the overall mood across asset classes. "
        "Paragraph 2: what stands out (most bullish / most bearish, any notable trend). "
        "Paragraph 3: what it means for a diversified systematic portfolio like Blackbox, "
        "in general educational terms (no advice, no buy/sell calls).\n\n"
        f"AI SENTIMENT READS:\n{chr(10).join(lines)}{an}"
    )
    text = ai_client.ask(prompt, system=GUARD, max_tokens=650)
    if not text:
        return None
    return {"text": text, "generated": sent.get("generated"),
            "n_scanned": sent.get("n_scanned"),
            "scores": {a: v.get("score") for a, v in assets.items()}}


# ---------------------------------------------------------------- tutor / coach
def coach(question, history=None):
    """General investing tutor grounded in the app's own strategies. Returns text or None."""
    if not ai_client.available() or not (question or "").strip():
        return None
    catalog = "\n".join(f"- {_stat_line(s)}" for s in _strategies())
    convo = ""
    if history:
        convo = "\n".join(f"{'User' if h.get('role')=='user' else 'You'}: {h.get('text','')}"
                          for h in history[-6:]) + "\n"
    prompt = (
        "You are teaching a curious beginner about systematic investing and about how "
        "THIS app's strategies work. Answer their question clearly in 2-5 sentences. If "
        "they ask about a specific strategy, ground it in the real facts below. If they "
        "ask a general concept, explain it simply. Never recommend what to buy or sell.\n\n"
        f"THE APP'S STRATEGIES:\n{catalog}\n\n{convo}User: {question.strip()[:500]}"
    )
    return ai_client.ask(prompt, system=GUARD, max_tokens=450)


# ---------------------------------------------------------------- glossary
def glossary(term):
    """Explain one investing term in plain English. Returns text or None."""
    if not ai_client.available() or not (term or "").strip():
        return None
    prompt = (
        f'Explain the investing/finance term "{term.strip()[:60]}" to a beginner in 2-4 '
        "sentences. Plain English, one concrete example if it helps. If it's a risk metric "
        "(like Sharpe ratio, max drawdown, beta, volatility), say what a good vs bad value "
        "looks like. No advice."
    )
    return ai_client.ask(prompt, system=GUARD, max_tokens=300)


# ---------------------------------------------------------------- strategy deep-dive
def strategy_deep_dive(sid):
    """Deep 'why this works' explanation for one strategy, from its real facts."""
    s = _strat(sid)
    if not s or not ai_client.available():
        return None
    st = s.get("stats", {})
    facts = (f"Name: {s['name']}. Tagline: {s.get('tagline','')}. Risk: {s.get('risk','')}. "
             f"CAGR {st.get('cagr')}%, Sharpe {st.get('sharpe')}, vol {st.get('vol')}%, "
             f"max drawdown {st.get('max_dd')}%, beta {st.get('beta')}, "
             f"Sharpe by decade {st.get('sharpe_h1')}/{st.get('sharpe_h2')}. "
             f"Methodology: {s.get('method','')}")
    prompt = (
        "Explain to a curious user WHY this strategy works and where its edge comes from, "
        "in 2 short paragraphs, using only the facts below. Cover: (1) the economic or "
        "behavioural reason it has an edge, and (2) what its risk profile means — read the "
        "Sharpe, drawdown and beta for them, and note honestly where it's weak. The "
        "'Sharpe by decade' shows whether the edge held in both halves of history — mention "
        "if it did. No advice, no buy/sell calls.\n\n"
        f"FACTS:\n{facts}"
    )
    return ai_client.ask(prompt, system=GUARD, max_tokens=550)


# ---------------------------------------------------------------- positioning desk
def positioning():
    """AI reads every strategy's LIVE rule-based stance + the LIVE AI sentiment
    overlay, and writes a per-strategy 'support note' explaining today's stance
    and how the AI risk overlay is (or isn't) adjusting it. This makes the
    existing AI->strategy support mechanism visible and explained. Returns
    {sid: {note, tilt, tilt_why, holdings, day_ret}} or None."""
    if not ai_client.available():
        return None
    try:
        from user_engine import marks
        mk = marks().get("strategies", {})
    except Exception:
        return None
    if not mk:
        return None
    names = {s["id"]: s["name"] for s in _strategies()}
    out = {}
    for sid, m in mk.items():
        holds = ", ".join(f"{h['name']} {h['w']}%" for h in m.get("holdings", [])) or "100% cash"
        scores = ", ".join(f"{b}:{v}" for b, v in (m.get("sentiment") or {}).items()) or "n/a"
        trimmed = m.get("tilt", 1.0) < 1.0
        facts = (f"Strategy: {names.get(sid, sid)}. Rule-based stance today: {m.get('note','')}. "
                 f"Currently holding: {holds}. Live AI sentiment on its assets: {scores} "
                 f"(scale -100..+100). AI risk overlay: "
                 + (f"ACTIVE — trimming exposure to {int(m.get('tilt',1)*100)}% because {m.get('tilt_why','')}"
                    if trimmed else "not trimming (sentiment not bearish enough to trigger)") + ".")
        prompt = (
            "In 2-3 sentences, explain this strategy's CURRENT positioning to the user and "
            "how the AI is supporting it right now. Make clear the position itself comes from "
            "the validated rule, and the AI sentiment overlay is a risk brake layered on top. "
            "Use only the facts. No advice.\n\n" + facts)
        note = ai_client.ask(prompt, system=GUARD, max_tokens=220)
        if note:
            out[sid] = {"name": names.get(sid, sid), "ai_note": note,
                        "tilt": m.get("tilt", 1.0), "tilt_why": m.get("tilt_why"),
                        "note": m.get("note", ""), "holdings": m.get("holdings", []),
                        "sentiment": m.get("sentiment", {})}
    return out or None


# ---------------------------------------------------------------- portfolio review
def portfolio_review(payload):
    """AI reviews the user's ACTUAL mix — diversification, concentration, risk —
    from the real per-strategy stats. payload is app_user._user_payload output.
    Returns text or None. Grounded only in real numbers."""
    if not ai_client.available():
        return None
    mix = payload.get("mix") or {}
    if not mix:
        return None
    cat = {s["id"]: s for s in _strategies()}
    lines = []
    cagr = vol = ddw = beta = tot = 0.0
    for sid, w in mix.items():
        s = cat.get(sid)
        if not s:
            continue
        st = s.get("stats", {})
        lines.append(f"- {w}% {s['name']} (risk {s.get('risk')}): CAGR {st.get('cagr')}%, "
                     f"vol {st.get('vol')}%, max drawdown {st.get('max_dd')}%, beta {st.get('beta')}")
        f = w / 100.0
        cagr += f * (st.get("cagr") or 0); vol += f * (st.get("vol") or 0)
        ddw += f * (st.get("max_dd") or 0); beta += f * (st.get("beta") or 0); tot += f
    # deterministic blended figures — the AI must use THESE, never compute its own
    blend_txt = (f"Blended (weighted) figures, already computed for you — use these exact "
                 f"numbers and do NOT calculate your own: CAGR {round(cagr,1)}%, volatility "
                 f"{round(vol,1)}%, weighted max drawdown {round(ddw,1)}%, weighted beta "
                 f"vs S&P {round(beta,2)}.")
    prompt = (
        "Review this user's actual portfolio mix in 2 short paragraphs, using ONLY the "
        "facts below. Never state a number that is not given to you; use the pre-computed "
        "blended figures verbatim and do not do any arithmetic yourself. Paragraph 1: how "
        "diversified vs concentrated it is, and what its blend of returns and risk looks "
        "like. Paragraph 2: read the weighted beta — is it low (independent from stocks) "
        "or high — and what that means for stability, plus be honest about any single-"
        "strategy concentration. No advice, no buy/sell calls.\n\n"
        f"MIX:\n{chr(10).join(lines)}\n\n{blend_txt}"
    )
    return ai_client.ask(prompt, system=GUARD, max_tokens=500)


# ---------------------------------------------------------------- compare
def compare(sid_a, sid_b):
    """AI verdict comparing two strategies, grounded in their real stats."""
    a, b = _strat(sid_a), _strat(sid_b)
    if not a or not b or not ai_client.available():
        return None
    prompt = (
        "Compare these two real, backtested strategies for the user in 2 short paragraphs, "
        "using only the facts below. Cover how they differ in return, risk (drawdown, "
        "volatility), and independence from the stock market (beta), and what kind of "
        "investor each suits in general educational terms. Note that combining low-beta, "
        "low-correlation strategies is how diversification lowers risk. No advice on which "
        "to pick, no buy/sell calls.\n\n"
        f"STRATEGY A — {_stat_line(a)}\n\nSTRATEGY B — {_stat_line(b)}"
    )
    return ai_client.ask(prompt, system=GUARD, max_tokens=550)
