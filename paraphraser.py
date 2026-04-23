"""
Text paraphrasing tool using NLTK WordNet and rule-based transformations.
No AI models — relies on POS tagging, synonym substitution, and sentence restructuring.
"""

import re
import random
from typing import Optional
from nltk.corpus import wordnet, stopwords, brown
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from textblob import TextBlob

STOP_WORDS = set(stopwords.words("english"))

# Brown corpus word frequencies — used to filter out uncommon synonym candidates
_BROWN_FREQ = FreqDist(w.lower() for w in brown.words())

# POS tags that map to WordNet POS constants
POS_MAP = {
    "JJ": wordnet.ADJ,
    "JJR": wordnet.ADJ,
    "JJS": wordnet.ADJ,
    "RB": wordnet.ADV,
    "RBR": wordnet.ADV,
    "RBS": wordnet.ADV,
    "NN": wordnet.NOUN,
    "NNS": wordnet.NOUN,
    "NNP": wordnet.NOUN,
    "NNPS": wordnet.NOUN,
    "VB": wordnet.VERB,
    "VBD": wordnet.VERB,
    "VBG": wordnet.VERB,
    "VBN": wordnet.VERB,
    "VBP": wordnet.VERB,
    "VBZ": wordnet.VERB,
}


def get_synonyms(word: str, pos: str, exclude: set[str]) -> list[str]:
    """
    Return synonyms for a word given its WordNet POS.
    Ranked by lemma frequency so common words are preferred over archaic ones.
    """
    wn_pos = POS_MAP.get(pos)
    if not wn_pos:
        return []

    orig_freq = _BROWN_FREQ[word.lower()]
    # Minimum frequency threshold: at least half as common as the original word,
    # and always at least 5 occurrences so obscure words are never chosen.
    min_freq = max(5, orig_freq // 2)

    scored: dict[str, int] = {}
    synsets = wordnet.synsets(word, pos=wn_pos)

    # Only consider the two most common senses to avoid archaic/obscure synonyms
    for syn in synsets[:2]:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            brown_freq = _BROWN_FREQ[name.lower()]
            if (
                name.lower() != word.lower()
                and name.lower() not in exclude
                and name.isalpha()
                and " " not in name
                and len(name) >= 3
                and brown_freq >= min_freq
            ):
                scored[name] = scored.get(name, 0) + brown_freq

    sorted_candidates = sorted(scored, key=lambda w: scored[w], reverse=True)
    return sorted_candidates[:6]


def preserve_case(original: str, replacement: str) -> str:
    """Match the capitalisation pattern of the original word."""
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement.lower()


def synonym_swap(sentence: str, intensity: float = 0.5) -> str:
    """Replace eligible words with synonyms at the given intensity (0–1)."""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    result = []

    for word, tag in tagged:
        if (
            word.lower() in STOP_WORDS
            or not word.isalpha()
            or tag not in POS_MAP
            or random.random() > intensity
        ):
            result.append(word)
            continue

        synonyms = get_synonyms(word.lower(), tag, {word.lower()})
        if synonyms:
            chosen = random.choice(synonyms)
            result.append(preserve_case(word, chosen))
        else:
            result.append(word)

    # Rejoin and fix spacing around punctuation
    text = " ".join(result)
    text = re.sub(r" ([,.!?;:])", r"\1", text)
    return text


def reorder_clauses(sentence: str) -> str:
    """
    Attempt basic clause reordering for sentences containing 'because', 'although',
    'since', 'while', 'if', 'when', 'after', 'before', 'unless'.
    Swaps subordinate clause to the front (or back if already at front).
    """
    subordinators = r"\b(because|although|since|while|if|when|after|before|unless)\b"
    match = re.search(subordinators, sentence, re.IGNORECASE)
    if not match:
        return sentence

    pivot = match.start()
    # Already starts with subordinator — move main clause first
    if pivot < 5:
        parts = re.split(subordinators, sentence, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 3:
            main_candidate = parts[0].strip(" ,")
            sub_clause = (parts[1] + parts[2]).strip()
            if main_candidate:
                return f"{sub_clause.capitalize()}, {main_candidate.lower()}."
        return sentence

    main = sentence[:pivot].strip().rstrip(",")
    sub = sentence[pivot:].strip().rstrip(".")
    return f"{sub.capitalize()}, {main.lower()}."


def restructure_sentence(sentence: str) -> str:
    """Apply light structural transformations to vary sentence form."""
    sentence = sentence.strip()

    # Expand common contractions
    contractions = {
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bshouldn't\b": "should not",
        r"\bwouldn't\b": "would not",
        r"\bcouldn't\b": "could not",
        r"\bI'm\b": "I am",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bit's\b": "it is",
        r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",
        r"\byou're\b": "you are",
        r"\bI've\b": "I have",
        r"\bwe've\b": "we have",
        r"\bthey've\b": "they have",
        r"\bI'd\b": "I would",
        r"\bhe'd\b": "he would",
        r"\bshe'd\b": "she would",
        r"\bthey'd\b": "they would",
    }
    for pattern, expansion in contractions.items():
        sentence = re.sub(pattern, expansion, sentence, flags=re.IGNORECASE)

    sentence = reorder_clauses(sentence)
    return sentence


def transition_swap(text: str) -> str:
    """Replace common transitional phrases with equivalent alternatives."""
    transitions = {
        r"\bIn addition\b": random.choice(["Furthermore", "Moreover", "Additionally"]),
        r"\bHowever\b": random.choice(["Nevertheless", "Nonetheless", "Yet"]),
        r"\bTherefore\b": random.choice(["Consequently", "As a result", "Thus"]),
        r"\bFor example\b": random.choice(["For instance", "To illustrate", "As an example"]),
        r"\bIn conclusion\b": random.choice(["To summarize", "In summary", "Ultimately"]),
        r"\bMoreover\b": random.choice(["Furthermore", "In addition", "Additionally"]),
        r"\bFurthermore\b": random.choice(["Moreover", "In addition", "Additionally"]),
        r"\bNevertheless\b": random.choice(["However", "Nonetheless", "Even so"]),
        r"\bAs a result\b": random.choice(["Therefore", "Consequently", "Thus"]),
        r"\bOn the other hand\b": random.choice(["Conversely", "In contrast", "Alternatively"]),
        r"\bIn other words\b": random.choice(["That is to say", "To put it differently", "Put simply"]),
        r"\bFirst of all\b": random.choice(["To begin with", "Initially", "First"]),
        r"\bIn particular\b": random.choice(["Specifically", "Notably", "Especially"]),
    }
    for pattern, replacement in transitions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def paraphrase(text: str, intensity: float = 0.5, variants: int = 1) -> list[str]:
    """
    Paraphrase input text.

    Args:
        text:      The input text to paraphrase.
        intensity: Fraction of eligible words to replace with synonyms (0.0–1.0).
        variants:  Number of different paraphrase variants to generate.

    Returns:
        A list of paraphrased strings.
    """
    intensity = max(0.0, min(1.0, intensity))
    results = []

    sentences = sent_tokenize(text)

    for _ in range(variants):
        paraphrased_sentences = []
        for sent in sentences:
            s = restructure_sentence(sent)
            s = synonym_swap(s, intensity)
            paraphrased_sentences.append(s)

        output = " ".join(paraphrased_sentences)
        output = transition_swap(output)

        # Ensure first character is capitalised
        if output:
            output = output[0].upper() + output[1:]

        results.append(output)

    return results


def readability_stats(text: str) -> dict:
    """Return basic readability metrics for the text."""
    blob = TextBlob(text)
    sentences = sent_tokenize(text)
    words = [w for w in word_tokenize(text) if w.isalpha()]

    avg_sent_len = len(words) / len(sentences) if sentences else 0
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0

    return {
        "sentences": len(sentences),
        "words": len(words),
        "avg_sentence_length": round(avg_sent_len, 1),
        "avg_word_length": round(avg_word_len, 1),
        "sentiment_polarity": round(blob.sentiment.polarity, 3),
        "sentiment_subjectivity": round(blob.sentiment.subjectivity, 3),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _prompt_multiline(prompt: str) -> str:
    """Read multi-line input until the user enters a blank line."""
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return " ".join(lines)


def cli() -> None:
    print()
    _print_separator("=")
    print("  TEXT PARAPHRASER  --  rule-based, no AI models")
    _print_separator("=")
    print()

    while True:
        text = _prompt_multiline(
            "Paste your text below (press Enter on a blank line when done):\n"
        )
        if not text.strip():
            print("No text entered. Please try again.\n")
            continue

        try:
            intensity_input = input("\nSynonym intensity (0.1 - 1.0, default 0.5): ").strip()
            intensity = float(intensity_input) if intensity_input else 0.5
        except ValueError:
            intensity = 0.5

        try:
            variants_input = input("Number of variants to generate (default 1): ").strip()
            variants = int(variants_input) if variants_input else 1
            variants = max(1, min(variants, 10))
        except ValueError:
            variants = 1

        print()
        _print_separator()
        print("ORIGINAL")
        _print_separator()
        print(text)
        stats_orig = readability_stats(text)
        print(f"\n[Stats] {stats_orig['sentences']} sentences | {stats_orig['words']} words | "
              f"avg sent len {stats_orig['avg_sentence_length']} | "
              f"avg word len {stats_orig['avg_word_length']}")

        results = paraphrase(text, intensity=intensity, variants=variants)

        for i, result in enumerate(results, 1):
            print()
            _print_separator()
            print(f"PARAPHRASE  (variant {i}/{variants})")
            _print_separator()
            print(result)
            stats_out = readability_stats(result)
            print(f"\n[Stats] {stats_out['sentences']} sentences | {stats_out['words']} words | "
                  f"avg sent len {stats_out['avg_sentence_length']} | "
                  f"avg word len {stats_out['avg_word_length']}")

        print()
        again = input("Paraphrase another text? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nDone.")


if __name__ == "__main__":
    cli()
