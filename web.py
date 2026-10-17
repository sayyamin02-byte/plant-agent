"""
web.py — a browser UI for the plant agent (Streamlit).

Type a plant problem (optionally upload a photo) and watch the agent use its
tools and return a structured diagnosis.

Run:
    pip install streamlit
    streamlit run web.py          # opens http://localhost:8501
"""
import io, os, tempfile, contextlib
import streamlit as st
from dotenv import load_dotenv
from agent import ask

load_dotenv()
st.set_page_config(page_title="Sprout — Plant Doctor", page_icon="🌱")

st.title("🌱 Sprout — Plant Care Agent")
st.caption("Ask a plant problem. The AI agent looks up care facts, checks today's "
           "live air quality, estimates watering — then gives a structured fix. "
           "Tuned for Delhi-NCR.")

with st.sidebar:
    st.subheader("Try asking")
    for q in ["My money plant's leaves are turning yellow in Delhi.",
              "Brown crispy tips on my plant in Noida, soil isn't soggy.",
              "Which low-light plant should a total beginner get?",
              "Tiny webs under the leaves this week — what is it?"]:
        st.markdown(f"- {q}")
    st.divider()
    vision = "on" if os.getenv("VISION_MODEL") else "off (set VISION_MODEL to enable)"
    st.caption(f"Photo diagnosis: **{vision}**. Needs a free LLM key in `.env`.")

q = st.chat_input("Describe your plant problem…")
photo = st.file_uploader("Optional: upload a plant photo", type=["jpg", "jpeg", "png"])

if q:
    st.chat_message("user").markdown(q + ("  📷" if photo else ""))

    img_path = None
    if photo:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(photo.getbuffer()); tmp.close(); img_path = tmp.name

    with st.chat_message("assistant"):
        with st.spinner("Thinking & using tools…"):
            steps = io.StringIO()
            with contextlib.redirect_stdout(steps):            # capture the agent's steps
                result = ask(q, image_path=img_path, verbose=True)

        # The structured answer (this is the "structured output" skill)
        st.markdown(f"### 🌿 {result.get('diagnosis','—')}")
        if result.get("why"):
            st.markdown(f"**Why:** {result['why']}")
        st.markdown(f"**Fix:** {result.get('fix','—')}")
        c1, c2 = st.columns(2)
        c1.metric("Confidence", str(result.get("confidence", "—")).title())
        c2.metric("Used live AQI", "Yes" if result.get("used_air_quality") else "No")

        with st.expander("🔧 how the agent worked (tools it used)"):
            st.code(steps.getvalue() or "(no steps captured)", language="text")
