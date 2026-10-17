"""
agent.py — the "brain": an agent loop.

The difference between a chatbot and an AGENT: a chatbot answers in one shot; an
agent can take SEVERAL turns — it thinks, calls a tool, reads the result, thinks
again, calls another tool — until it's ready to answer. This file runs that loop.

Flow:
  1. We send the question + the list of tools to the model.
  2. If the model replies with a tool call, we run that Python tool and send the
     result back. (repeat)
  3. When the model calls "submit_answer", we stop and return that structured result.

Also optional: if you pass an image, we first ask a VISION model to describe the
plant's visible symptoms, and feed that description into the agent.
"""
from __future__ import annotations
import os, json, base64
from openai import OpenAI
from dotenv import load_dotenv
import tools

load_dotenv()

def _client():
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise SystemExit("Set LLM_API_KEY in .env (free Groq key: https://console.groq.com).")
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))

SYSTEM = (
    "You are 'Sprout', a friendly plant-care assistant for Delhi-NCR homes. "
    "Use the tools to look up care facts, check live air quality when dust/pollution is relevant, "
    "and estimate watering. Reason step by step, then call submit_answer with a clear, honest fix. "
    "Be specific to NCR conditions (hard water, dust, heat, monsoon). Do not invent facts — "
    "if unsure, say so and keep confidence low."
)

def describe_image(path: str) -> str:
    """OPTIONAL: ask a vision model what's visibly wrong with the plant in the photo."""
    model = os.getenv("VISION_MODEL")
    if not model:
        return ""  # vision is optional; skip if no vision model configured
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    msg = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this plant and any visible problems "
         "(leaf colour, spots, drooping, pests) in 2-3 factual sentences."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    out = _client().chat.completions.create(model=model, temperature=0, messages=msg)
    return out.choices[0].message.content.strip()

def ask(question: str, image_path: str | None = None, max_steps: int = 6, verbose: bool = True) -> dict:
    client = _client()
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

    if image_path:
        seen = describe_image(image_path)
        if seen:
            question = f"{question}\n\n[What the photo shows: {seen}]"
            if verbose: print(f"👁️  photo: {seen}\n")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=messages, tools=tools.TOOL_SCHEMAS)
        m = resp.choices[0].message
        messages.append(m.model_dump(exclude_none=True))      # remember what the model said

        if not m.tool_calls:                                  # plain text, no tool -> done
            return {"diagnosis": m.content, "fix": "", "confidence": "medium",
                    "used_air_quality": False, "answer_text": m.content}

        for call in m.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            if name == "submit_answer":                       # the FINAL structured answer
                if verbose: print(f"✅ step {step+1}: final answer\n")
                return args
            result = tools.run_tool(name, args)               # run a real tool
            if verbose: print(f"🔧 step {step+1}: {name}({args}) -> {result[:90]}...")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return {"diagnosis": "Ran out of steps.", "fix": "Try rephrasing.", "confidence": "low"}
