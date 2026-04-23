# AI-Bypasser

A Python and Streamlit-based text rewriting tool that uses strict rule-based NLP (no AI models) to humanize and paraphrase text. 

Crafted with intention by **Shlok Tilokani** — because your words deserve to sound like yours.

## Acknowledgments

This service was built using methods and inspiration from this YouTube tutorial: [https://youtu.be/LDEBs9Qw1aU](https://youtu.be/LDEBs9Qw1aU)

## Features

This project implements the **Dr. Marcin Kriukow methodology** for humanizing AI-generated content through an 8-step pipeline:

1. **Synonym Swapping**: Uses WordNet and filters candidates by Brown corpus frequency to avoid archaic or unnatural words.
2. **Contraction Expansion**: Replaces formal phrasing with natural contractions.
3. **Filler Phrase Removal**: Strips out common AI filler phrases (e.g., "It is important to note that").
4. **Hedging Injection**: Adds natural uncertainty or nuance.
5. **Transition Diversification**: Swaps repetitive transitions with a varied extended map (e.g., "However" → "That said").
6. **Sentence-Opening Variation**: Modifies repetitive sentence structures.
7. **Burstiness Injection**: Splits excessively long sentences and merges short ones to mimic human variability in sentence length.
8. **Clause Restructuring**: Inverts and fronts clause structures.

The tool provides a dual-tab Streamlit UI featuring a **Synonym Engine** and the full **Humanizer Pipeline**, with adjustable intensity controls.

## Tech Stack

- **Python**
- **Streamlit** (User Interface)
- **NLTK & TextBlob** (NLP processing)

## Local Setup

1. **Activate your virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required NLTK corpora**:
   The script relies on several datasets. Download them by running:
   ```bash
   python3 -c "import nltk; import ssl; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download(['wordnet', 'averaged_perceptron_tagger', 'punkt', 'punkt_tab', 'omw-1.4', 'stopwords', 'brown'])"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Deployment

This application is containerized with Docker and configured for automated deployment on **GCP Cloud Run** (via GitHub and Cloud Build triggers). 

---

## Disclaimer

**IMPORTANT:** This tool is a rule-based writing aid designed to help refine, diversify, and restructure text. All core ideas, arguments, and research must originate with the user. This tool is **not** intended for fabricating authorship, academic misconduct, or bypassing integrity systems maliciously. The user bears full and complete responsibility for how the generated content is utilized, represented, and submitted.
