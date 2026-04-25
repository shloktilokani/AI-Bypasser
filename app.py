import streamlit as st
import streamlit.components.v1 as components
from paraphraser import paraphrase
from humanizer import humanize

st.set_page_config(page_title="AI Bypasser", layout="wide")

st.title("AI Bypasser")
st.caption("Rule-based text rewriting. No AI models.")
st.caption("Crafted with intention by **Shlok Tilokani** — because your thoughts deserve to sound like yours.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    intensity = st.slider(
        "Synonym intensity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    st.caption(
        "Controls how many words get replaced with synonyms. "
        "Low values (0.1–0.3) make minimal changes and keep the original phrasing mostly intact. "
        "High values (0.7–1.0) replace most eligible words but can produce unnatural or grammatically awkward output. "
        "\n\n**Recommended sweet spot: 0.4–0.5** — enough rewording to meaningfully paraphrase while keeping the text fluent and readable."
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "syn_result" not in st.session_state:
    st.session_state.syn_result = ""
if "hum_result" not in st.session_state:
    st.session_state.hum_result = ""

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_syn, tab_hum = st.tabs(["Paraphraser", "Humanizer"])


def _copy_button(text: str) -> None:
    """Render a green copy-to-clipboard button matching Streamlit's button style."""
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n")
    components.html(
        f"""
        <button
            onclick="navigator.clipboard.writeText(`{safe}`).then(()=>{{
                this.innerText='Copied!';
                setTimeout(()=>this.innerText='Copy to clipboard', 2000);
            }})"
            style="
                display:inline-flex; align-items:center; justify-content:center;
                font-weight:400; padding:0.25rem 0.75rem; border-radius:0.5rem;
                min-height:2.4rem; font-size:1rem; line-height:1.6;
                color:white; background-color:#28a745;
                border:1px solid transparent; cursor:pointer;
                font-family:'Source Sans Pro',sans-serif;
            "
        >Copy to clipboard</button>
        """,
        height=50,
    )


# ── Tab 1: Paraphraser ───────────────────────────────────────────────────────────
with tab_syn:
    col_in, col_out = st.columns(2, gap="medium")

    with col_in:
        st.subheader("Input")
        syn_input = st.text_area(
            label="syn_input",
            placeholder="Paste your text here...",
            height=380,
            label_visibility="collapsed",
            key="syn_input",
        )
        syn_clicked = st.button(
            "Paraphrase",
            type="primary",
            disabled=not syn_input.strip(),
            key="syn_btn",
        )

    if syn_clicked and syn_input.strip():
        try:
            with st.spinner("Paraphrasing..."):
                results = paraphrase(
                    syn_input.strip(), intensity=float(intensity), variants=1
                )
                st.session_state.syn_result = results[0]
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.syn_result = ""

    with col_out:
        st.subheader("Output")
        st.text_area(
            label="syn_output",
            value=st.session_state.syn_result,
            placeholder="Paraphrased text will appear here...",
            height=380,
            label_visibility="collapsed",
        )
        if st.session_state.syn_result:
            _copy_button(st.session_state.syn_result)


# ── Tab 2: Humanizer ──────────────────────────────────────────────────────────
with tab_hum:
    col_in2, col_out2 = st.columns(2, gap="medium")

    with col_in2:
        st.subheader("Input")
        hum_input = st.text_area(
            label="hum_input",
            placeholder="Paste AI-generated text here...",
            height=380,
            label_visibility="collapsed",
            key="hum_input",
        )
        hum_clicked = st.button(
            "Humanize",
            type="primary",
            disabled=not hum_input.strip(),
            key="hum_btn",
        )

    if hum_clicked and hum_input.strip():
        try:
            with st.spinner("Humanizing..."):
                st.session_state.hum_result = humanize(
                    hum_input.strip(), intensity=float(intensity)
                )
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.hum_result = ""

    with col_out2:
        st.subheader("Output")
        st.text_area(
            label="hum_output",
            value=st.session_state.hum_result,
            placeholder="Humanized text will appear here...",
            height=380,
            label_visibility="collapsed",
        )
        if st.session_state.hum_result:
            _copy_button(st.session_state.hum_result)

st.markdown("---")
st.caption(
    "**Disclaimer:** This tool is a rule-based writing aid — it helps give clearer form to ideas that are already yours. "
    "The words may be restructured, but the thinking, arguments, and intellectual substance must originate with you. "
    "This tool is not designed to fabricate ideas, misrepresent authorship, or circumvent academic integrity standards. "
    "By using it, you accept full responsibility for the honesty, accuracy, and originality of the content you submit and publish."
)
