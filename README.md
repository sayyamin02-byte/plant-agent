# 🌱 Sprout — an Agentic Plant-Care Assistant (Delhi-NCR)

A small, open-source AI project that diagnoses houseplant problems. You ask a
question (optionally with a photo), and an AI **agent** figures out the answer by
*using tools* — looking up care facts, checking today's real air quality, and
estimating a watering schedule — then returns a clear, structured fix.

Built as a learning project to go one step beyond a basic RAG chatbot.

---

## What makes this different from a normal chatbot

A chatbot answers in **one shot** from what it already knows. An **agent** can take
several turns: it thinks, uses a tool, reads the result, thinks again, and only then
answers. This project teaches three skills a plain RAG bot doesn't:

1. **Tool use + an agent loop** — the AI decides *when* to call a function.
2. **A real live-data tool** — it fetches today's actual air quality from a free API.
3. **Vision (optional)** — it can look at a *photo* of your plant and describe what's wrong.
4. **Structured output** — the final answer comes back as clean fields (diagnosis, fix, confidence), not a blob of text.
5. **Evaluation** — a tiny test suite that *measures* whether the answers are actually good.

---

## How it works, in plain words

Think of the AI as a smart intern who can only **talk**, so we give it a few buttons
("tools") it can press:

- 🔎 **search_plant_care** — press this to look up care facts from our knowledge file. *(This is RAG: it finds the paragraphs most related to your question by meaning, not keywords.)*
- 🌫️ **get_air_quality** — press this to check today's real AQI/PM2.5 for your city (e.g. if dust might be the problem). *(A real internet call — free, no key.)*
- 💧 **watering_plan** — press this to estimate how often to water. *(Just a Python function — a tool doesn't have to be an API.)*
- ✅ **submit_answer** — press this when ready, to hand back the final answer as structured fields.

The **agent loop** (`agent.py`) is simply: *ask the AI → if it pressed a tool button,
run that tool and give it the result → repeat → when it presses `submit_answer`, stop.*

If you add `--image photo.jpg` and set a `VISION_MODEL`, a vision model first looks at
the photo and writes down what it sees (yellow leaves, webbing, etc.), and that
description is fed into the agent — so it can "see" your plant.

---

## Setup (5 minutes)

```bash
cd plant-agent
python -m venv .venv && . .venv/Scripts/activate      # Windows (use source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
cp .env.example .env            # then open .env and paste a free Groq key
```
Get a **free** LLM key at <https://console.groq.com>. Vision is optional (leave `VISION_MODEL` blank to skip it).

## Run it

```bash
# ask a question
python app.py "my money plant leaves are turning yellow in Delhi"

# with a photo (needs VISION_MODEL set)
python app.py "white dusty film on the leaves, should I worry?" --image photo.jpg

# measure answer quality
python eval.py
```

You'll see the agent's steps (which tools it used), then a structured diagnosis + fix.

---

## Project structure

```
plant-agent/
├── app.py            # command-line entry point
├── agent.py          # the agent loop (the "brain") + optional vision
├── tools.py          # the 4 tools: RAG search, air quality, watering, submit_answer
├── eval.py           # a small test suite that scores the answers
├── data/
│   └── plant_care.md # the knowledge base the RAG tool searches
├── requirements.txt
└── .env.example
```

## What to try next (to keep learning)

- Add a new tool (e.g. a weather/temperature lookup) and watch the agent use it.
- Add more test cases to `eval.py`, then try different `LLM_MODEL`s and compare scores.
- Expand `data/plant_care.md` and see retrieval improve.
- Swap in a stronger vision model and test real photos.

---

*A personal learning project. General plant-care knowledge only — not affiliated with
or using any company's private data.*
