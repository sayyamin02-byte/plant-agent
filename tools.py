"""
tools.py — the "hands" of the agent.

An AI model can only talk. TOOLS are small Python functions we let it call so it
can actually DO things: look up care facts, check today's air quality, or do a
calculation. We describe each tool to the model (TOOL_SCHEMAS), the model decides
when to call one, and run_tool() runs the real Python function.

Three tools here:
  1. search_plant_care  -> RAG: find relevant lines in our care knowledge base
  2. get_air_quality    -> a REAL API call (free, no key) for live AQI/PM2.5
  3. watering_plan      -> a plain rule-based calculator (tools don't have to be APIs)
"""
from __future__ import annotations
import json, os, math, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — search_plant_care  (this is the "RAG" part: Retrieve relevant text)
# ─────────────────────────────────────────────────────────────────────────────
# We load the knowledge base once, turn each paragraph into a vector (a list of
# numbers that captures its meaning), and at query time return the paragraphs
# whose vectors are closest to the question's vector. That's all RAG is.
_CHUNKS: list[str] = []
_VECS = None
_MODEL = None

def _load_kb():
    global _CHUNKS, _VECS, _MODEL
    if _VECS is not None:
        return
    from sentence_transformers import SentenceTransformer  # local, free embeddings
    import numpy as np
    text = open(os.path.join(HERE, "data", "plant_care.md"), encoding="utf-8").read()
    # split into paragraphs (chunks); keep only non-trivial ones
    _CHUNKS = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _VECS = _MODEL.encode(_CHUNKS, normalize_embeddings=True)
    _VECS = np.asarray(_VECS)

def search_plant_care(query: str, k: int = 3) -> str:
    """Return the k most relevant care snippets for the query."""
    import numpy as np
    _load_kb()
    qv = _MODEL.encode([query], normalize_embeddings=True)[0]
    scores = _VECS @ qv                      # cosine similarity (vectors are normalized)
    top = np.argsort(scores)[::-1][:k]
    return "\n---\n".join(_CHUNKS[i] for i in top)

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — get_air_quality  (a REAL external API; free, no key: open-meteo)
# ─────────────────────────────────────────────────────────────────────────────
_CITIES = {
    "delhi": (28.61, 77.21), "noida": (28.54, 77.39), "gurgaon": (28.46, 77.03),
    "gurugram": (28.46, 77.03), "ghaziabad": (28.67, 77.45), "faridabad": (28.41, 77.31),
}

def get_air_quality(city: str = "delhi") -> str:
    """Live air quality (US AQI + PM2.5) for an NCR city, from open-meteo (no key)."""
    lat, lon = _CITIES.get(city.strip().lower(), _CITIES["delhi"])
    url = ("https://air-quality-api.open-meteo.com/v1/air-quality?"
           + urllib.parse.urlencode({"latitude": lat, "longitude": lon,
                                     "current": "us_aqi,pm2_5,pm10", "timezone": "auto"}))
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            cur = json.load(r).get("current", {})
        return (f"{city.title()} right now — US AQI {cur.get('us_aqi','?')}, "
                f"PM2.5 {cur.get('pm2_5','?')} µg/m³, PM10 {cur.get('pm10','?')} µg/m³.")
    except Exception as e:
        return f"(could not fetch air quality: {e})"

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — watering_plan  (a plain function; shows a tool can just be local logic)
# ─────────────────────────────────────────────────────────────────────────────
def watering_plan(plant_type: str = "", pot_inches: float = 6, season: str = "summer") -> str:
    base = {"summer": 4, "monsoon": 10, "winter": 12, "spring": 6}.get(season.lower(), 7)
    if pot_inches >= 10: base += 2            # bigger pot holds water longer
    thirsty = any(w in plant_type.lower() for w in ("fern", "peace lily", "calathea"))
    hardy   = any(w in plant_type.lower() for w in ("snake", "zz", "cact", "succulent"))
    if thirsty: base = max(2, base - 2)
    if hardy:   base += 4
    return (f"Water about every {base} days (finger-test first: water only if the top "
            f"2 inches are dry). Season={season}, pot={pot_inches}\" ({plant_type or 'plant'}).")

# ─────────────────────────────────────────────────────────────────────────────
# Tell the model what tools exist and how to call them (OpenAI "tools" format).
# The 4th "tool" is how the model hands back its FINAL structured answer.
# ─────────────────────────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_plant_care",
        "description": "Look up plant-care facts (watering, light, hard water, pests) from the knowledge base.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_air_quality",
        "description": "Get today's live air quality (AQI, PM2.5) for a Delhi-NCR city. Use when dust/pollution may matter.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "delhi, noida, gurgaon, ghaziabad or faridabad"}}}}},
    {"type": "function", "function": {
        "name": "watering_plan",
        "description": "Estimate how often to water, given plant type, pot size (inches) and season.",
        "parameters": {"type": "object", "properties": {
            "plant_type": {"type": "string"}, "pot_inches": {"type": "number"},
            "season": {"type": "string", "description": "summer, monsoon, winter or spring"}}}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Give the FINAL answer as structured data once you have enough info.",
        "parameters": {"type": "object", "properties": {
            "diagnosis":   {"type": "string", "description": "the most likely issue in one line"},
            "why":         {"type": "string", "description": "the reasoning / evidence"},
            "fix":         {"type": "string", "description": "clear step-by-step fix"},
            "confidence":  {"type": "string", "enum": ["low", "medium", "high"]},
            "used_air_quality": {"type": "boolean"}},
            "required": ["diagnosis", "fix", "confidence"]}}},
]

# Map tool name -> real python function (submit_answer is handled by the agent loop).
_DISPATCH = {"search_plant_care": search_plant_care,
             "get_air_quality": get_air_quality,
             "watering_plan": watering_plan}

def run_tool(name: str, args: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"(unknown tool: {name})"
    try:
        return str(fn(**args))
    except Exception as e:
        return f"(tool {name} error: {e})"
