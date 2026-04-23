# Project Memory

## Project Overview
A Python + Streamlit text rewriting tool using rule-based NLP only (no AI models). Located at `d:/AI_Plagrism_Bypasser/`.
Guided by the Dr. Marcin Kriukow methodology stored in `.context/humanizing-ai-content-guide.md`.

## File Structure
```
d:/AI_Plagrism_Bypasser/
├── app.py               # Streamlit UI — two tabs: Synonyms + Humanizer
├── paraphraser.py       # Synonym swapping engine (WordNet + Brown corpus freq filter)
├── humanizer.py         # Full humanization pipeline (8-step Kriukow method)
├── requirements.txt     # streamlit, nltk, textblob
├── Dockerfile           # python:3.12-slim, NLTK corpora downloaded at build time
├── cloudbuild.yaml      # GCP Cloud Build → Artifact Registry → Cloud Run
├── .dockerignore
└── .context/
    └── humanizing-ai-content-guide.md
```

## Runtime
- Local: Python 3.14.2, venv at `.venv/`, Streamlit on port 8502
- Docker: python:3.12-slim, port 8080 (Cloud Run injects $PORT)
- Run locally: `.venv/Scripts/streamlit.exe run app.py`

## NLTK Corpora Required
wordnet, averaged_perceptron_tagger_eng, punkt_tab, omw-1.4, stopwords, brown

## Deployment
GCP Cloud Run via GitHub + Cloud Build trigger. Image in Artifact Registry.
Default region: `us-central1`, service: `text-paraphraser`, repo: `paraphraser`.

---

## paraphraser.py — Synonym Engine
- POS-tags tokens with NLTK, looks up WordNet synsets (first 2 only to avoid archaic words)
- Filters candidates by Brown corpus frequency: `min_freq = max(5, orig_freq // 2)`
- Returns top 6 by Brown frequency
- Also does: contraction expansion, clause reordering, transition swap
- Intensity controls fraction of eligible words replaced (0.1–1.0)

## humanizer.py — Kriukow Pipeline
Copy of paraphraser.py extended with the full 8-step methodology. Default intensity 0.3.

Pipeline order inside `humanize()`:
1. `_expand_contractions()` — 26 patterns
2. `remove_filler()` — strips 14 AI filler phrases ("It is important to note that", etc.)
3. Per-sentence: `inject_hedging()` → `vary_opening()` → `synonym_swap()`
4. `inject_burstiness()` — splits sentences >35 words, merges consecutive sentences <6 words
5. `diversify_transitions()` — 17-entry extended map ("However" → "That said / Yet / Even so")

What AI detectors measure (from the guide):
- **Perplexity** — AI picks safest next word; disrupt with unpredictable word choices
- **Burstiness** — AI has uniform sentence lengths; vary short and long
- **Syntactic consistency** — AI reuses same grammatical skeletons
- **Vocabulary profile** — AI uses safe mid-frequency words only

---

## app.py — Streamlit UI
- Wide layout, sidebar with intensity slider (0.1–1.0)
- Two tabs: `st.tabs(["Synonyms", "Humanizer"])`
- Each tab: `st.columns(2, gap="medium")` — left = input + button, right = output + copy button
- Variants hardcoded to 1 (user explicitly removed the variants control)
- Copy button: `components.html` injected HTML button calling `navigator.clipboard.writeText()`
- Sidebar has a `st.caption()` below the slider explaining the range and recommending 0.4–0.5 as the sweet spot
- Ownership credit on its own line below subtitle: two separate `st.caption()` calls — first for the tool description, second for "Crafted with intention by **Shlok Tilokani** — because your words deserve to sound like yours."

Copy button CSS (matches Streamlit native button):
```
padding:0.25rem 0.75rem; border-radius:0.5rem; min-height:2.4rem;
font-size:1rem; line-height:1.6; background-color:#28a745; color:white;
font-family:'Source Sans Pro',sans-serif;
```

- Disclaimer at bottom of page (after all tabs): rule-based writing aid, ideas must originate with user, not for fabricating authorship, user bears full responsibility

---

## Key Bug Fixes (do not regress)
1. **Output text_area key** — never use `key=` on the output text_area. Streamlit caches the empty value in session_state on first render and ignores `value=` on reruns.
2. **Button state** — capture `st.button()` return value directly (`clicked = st.button(...)`). `st.session_state.get("key")` does not work for button click detection.
3. **Windows encoding** — never use Unicode box-drawing chars (─ ═ · –) in print/Streamlit strings; causes `UnicodeEncodeError` on cp1252 terminals.
4. **Archaic synonyms** — the Brown corpus frequency filter prevents obscure WordNet lemmas ("hokey" for "artificial"). Do not remove this filter.
5. **Copy button placement** — always inside `with col_out:` directly below text_area, never in a separate `st.columns()` row (causes broken layout).

---

## User Preferences
- Concise responses, no verbose explanations
- Prefers to run commands himself — give the command, don't execute it
- Flags UI issues quickly; keep layout clean and consistent
- Works on Windows 11 with bash shell (Git Bash) and VSCode
- Email: asparmar@gulbrandsen.com
