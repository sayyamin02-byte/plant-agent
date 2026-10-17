"""
eval.py — does the assistant actually give good answers?

"It replied" is not the same as "it was right." An eval is a small set of test
questions with what a GOOD answer should contain, run automatically so you can
measure quality and catch regressions when you change a prompt or a tool.

Here we use a simple keyword check: for each test, does the answer mention the
ideas it should? Real teams use fancier scoring, but the idea is identical —
compare the model's output against expectations and report a score.

Run:  python eval.py
"""
from agent import ask

# question -> words we'd expect a correct answer/fix to mention (any of each group)
TESTS = [
    ("My money plant has soft yellow leaves and the soil is always wet.",
     [["overwater", "too much water", "soggy", "root rot"], ["stop", "let", "dry"]]),
    ("Brown crispy tips on my plant in Noida, soil isn't soggy.",
     [["hard water", "minerals", "tds", "filtered", "ro "]]),
    ("Which plant should a total beginner with a dark room get?",
     [["snake", "zz", "money", "pothos", "aglaonema"]]),
    ("Tiny webs under the leaves in winter — what is it?",
     [["spider mite", "mite"], ["neem", "wipe", "isolate"]]),
]

def covers(text: str, groups) -> bool:
    t = (text or "").lower()
    return all(any(w in t for w in group) for group in groups)

def main():
    passed = 0
    for i, (q, groups) in enumerate(TESTS, 1):
        r = ask(q, verbose=False)
        blob = " ".join(str(r.get(k, "")) for k in ("diagnosis", "why", "fix"))
        ok = covers(blob, groups)
        passed += ok
        print(f"{'✅' if ok else '❌'}  Test {i}: {q}")
        if not ok:
            print(f"     got: {blob[:120]}...")
    print(f"\nScore: {passed}/{len(TESTS)} passed")

if __name__ == "__main__":
    main()
