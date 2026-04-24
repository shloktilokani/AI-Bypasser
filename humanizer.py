"""
Humanizer — rule-based AI-text humanization.
Implements the Dr. Marcin Kriukow methodology:
  1. Synonym swapping (inherited from paraphraser)
  2. Contraction expansion
  3. Filler phrase removal
  4. Hedging injection
  5. Transition diversification (extended set)
  6. Sentence-opening variation
  7. Burstiness injection (sentence length variation)
  8. Inverted / fronted clause structures
No AI models — NLTK + TextBlob only.
"""

import re
import random
from nltk.corpus import wordnet, stopwords, brown
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
STOP_WORDS = set(stopwords.words("english"))
_BROWN_FREQ = FreqDist(w.lower() for w in brown.words())

POS_MAP = {
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV,
    "NN": wordnet.NOUN, "NNS": wordnet.NOUN, "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN,
    "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB,
    "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
}

# ── Filler phrases (Step 5 of guide) ─────────────────────────────────────────
FILLER_PATTERNS = [
    (r"It is important to (consider|note|recognise|recognize) (that )?", ""),
    (r"It is worth noting that ", ""),
    (r"It is crucial to (understand|acknowledge) (that )?", ""),
    (r"[Tt]his (is a )?(complex|complicated) issue (that )?requires? (a )?nuanced analysis\.?", ""),
    (r"[Tt]his has significant implications for society\.?", ""),
    (r"[Ff]urther research is needed\.?", ""),
    (r"[Ii]t goes without saying that ", ""),
    (r"[Nn]eedless to say,? ", ""),
    (r"[Ii]t is (widely |generally )?accepted that ", ""),
    (r"[Aa]s (we |one )?can see,? ", ""),
    (r"[Ii]n today's (society|world|day and age),? ", "In contemporary contexts, "),
    (r"[Ii]n (the )?conclusion,? it is (clear|evident|apparent) that ", ""),
    (r"[Ii]t (should be|must be|can be) (noted|mentioned|emphasised|emphasized) that ", ""),
    (r"[Aa]ll (things|factors|aspects) considered,? ", ""),
    (r"[Tt]o (summarize|summarise|conclude|sum up),? ", ""),
]

# ── Hedging maps (Step 2 of guide) ───────────────────────────────────────────
# Pattern: (regex matching absolute verb phrase, list of hedged replacements)
HEDGE_PATTERNS = [
    (r"\b(has|have) a (significant |direct |clear |major )?(impact|effect|influence) on\b",
     ["appears to influence", "may have an effect on", "has been shown to affect"]),
    (r"\b(causes?|results? in|leads? to)\b",
     ["may contribute to", "appears to result in", "has been associated with"]),
    (r"\b(proves?|demonstrates?|shows?)\b",
     ["suggests", "indicates", "appears to show"]),
    (r"\b(is|are) (clearly|obviously|undoubtedly|certainly)\b",
     ["appears to be", "is arguably", "may be"]),
    (r"\b(will|must) (always|inevitably)\b",
     ["tends to", "may", "is likely to"]),
    (r"\b(is|are) essential\b",
     ["appears to be important", "may be considered essential", "is often regarded as necessary"]),
    (r"\b(is|are) (the )?key\b",
     ["appears to be significant", "may be central", "is arguably important"]),
    (r"\b(is|are) (the )?(main|primary|sole) (cause|reason|factor)\b",
     ["may be a contributing factor", "appears to be one of the primary reasons",
      "has been identified as a key factor"]),
]

# ── Extended transition map (Step 6 of guide) ────────────────────────────────
TRANSITIONS = {
    r"\bHowever\b":          ["That said", "Yet", "Even so", "In contrast", "Interestingly"],
    r"\bTherefore\b":        ["As a result", "Consequently", "This suggests that", "Given this"],
    r"\bFurthermore\b":      ["Beyond this", "Adding to this", "This is compounded by", "Moreover"],
    r"\bMoreover\b":         ["Beyond this", "This also", "Relatedly", "A further consideration is"],
    r"\bIn addition\b":      ["This also", "Relatedly", "Beyond this", "Additionally"],
    r"\bAdditionally\b":     ["This also", "Relatedly", "A further point is", "Beyond this"],
    r"\bIn conclusion\b":    ["Taken together", "On balance", "When considered as a whole"],
    r"\bTo conclude\b":      ["On balance", "Taken together", "When considered collectively"],
    r"\bFor example\b":      ["For instance", "To illustrate", "As one example", "Consider"],
    r"\bFor instance\b":     ["As an example", "To illustrate", "Consider the case of"],
    r"\bNevertheless\b":     ["Even so", "That said", "Notwithstanding this"],
    r"\bNonetheless\b":      ["Even so", "That said", "Despite this"],
    r"\bAs a result\b":      ["Consequently", "This suggests that", "Given this", "Thus"],
    r"\bIn other words\b":   ["Put differently", "To put it another way", "That is"],
    r"\bOn the other hand\b":["Conversely", "In contrast", "By contrast"],
    r"\bIt is important to note\b": ["Notably", "Significantly", "Of particular relevance"],
    r"\bFirst of all\b":     ["To begin with", "Initially", "At the outset"],
    r"\bIn particular\b":    ["Specifically", "Notably", "Of particular note"],
}

# ── Sentence-opening variation patterns (Step 3 of guide) ────────────────────
# Each entry: (regex to match opening, builder fn or replacement string)
OPENING_INVERSIONS = [
    # "X is central/key/essential to Y" → "Central to Y is X"
    (r"^(.+?)\s+is\s+(central|key|essential|crucial|fundamental|vital)\s+to\s+(.+)\.$",
     lambda m: f"{m.group(2).capitalize()} to {m.group(3)} is {m.group(1)}."),
    # "X is the main/primary reason why Y" → "The main reason why Y is X"
    (r"^(.+?)\s+is\s+the\s+(main|primary|principal)\s+reason\s+(why|that)\s+(.+)\.$",
     lambda m: f"The {m.group(2)} reason {m.group(3)} {m.group(4)} is {m.group(1)}."),
    # "This study examines/shows/analyses X" → "Examining X, this study..."
    (r"^(This study|This paper|This research|This work)\s+(examines?|shows?|analyses?|analyzes?|investigates?|explores?)\s+(.+)\.$",
     lambda m: f"{m.group(2).capitalize().rstrip('s')}ing {m.group(3)}, {m.group(1).lower()} {m.group(2)} the extent to which this holds."),
]

# Subordinating conjunctions used for clause-fronting
SUBORDINATORS = r"\b(because|although|since|while|if|when|after|before|unless|whereas|given that)\b"


# ── Core helpers (shared with paraphraser) ───────────────────────────────────

def _get_synonyms(word: str, pos: str) -> list[str]:
    wn_pos = POS_MAP.get(pos)
    if not wn_pos:
        return []
    orig_freq = _BROWN_FREQ[word.lower()]
    min_freq = max(5, orig_freq // 2)
    scored: dict[str, int] = {}
    for syn in wordnet.synsets(word, pos=wn_pos)[:2]:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            freq = _BROWN_FREQ[name.lower()]
            if (
                name.lower() != word.lower()
                and name.isalpha()
                and " " not in name
                and len(name) >= 3
                and freq >= min_freq
            ):
                scored[name] = scored.get(name, 0) + freq
    return sorted(scored, key=lambda w: scored[w], reverse=True)[:6]


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement.lower()


def _expand_contractions(text: str) -> str:
    contractions = {
        r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
        r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bisn't\b": "is not",
        r"\baren't\b": "are not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
        r"\bhasn't\b": "has not", r"\bhaven't\b": "have not", r"\bhadn't\b": "had not",
        r"\bshouldn't\b": "should not", r"\bwouldn't\b": "would not", r"\bcouldn't\b": "could not",
        r"\bI'm\b": "I am", r"\bhe's\b": "he is", r"\bshe's\b": "she is", r"\bit's\b": "it is",
        r"\bthey're\b": "they are", r"\bwe're\b": "we are", r"\byou're\b": "you are",
        r"\bI've\b": "I have", r"\bwe've\b": "we have", r"\bthey've\b": "they have",
        r"\bI'd\b": "I would", r"\bhe'd\b": "he would", r"\bshe'd\b": "she would",
        r"\bthey'd\b": "they would",
    }
    for pattern, expansion in contractions.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


# ── Step 5: Filler removal ────────────────────────────────────────────────────

def remove_filler(text: str) -> str:
    """Strip known AI filler phrases from text."""
    for pattern, replacement in FILLER_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Collapse mid-line double spaces only — preserve leading indentation
    text = re.sub(r"(?m)(?<=\S)  +", " ", text)
    return text


# ── Step 2: Hedging injection ─────────────────────────────────────────────────

def inject_hedging(sentence: str) -> str:
    """Replace absolute-sounding verb phrases with appropriately hedged alternatives."""
    for pattern, replacements in HEDGE_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            replacement = random.choice(replacements)
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE, count=1)
            break  # one hedge per sentence to avoid over-editing
    return sentence


# ── Step 6: Transition diversification ───────────────────────────────────────

def diversify_transitions(text: str) -> str:
    """Replace overused AI transition words with varied alternatives."""
    for pattern, options in TRANSITIONS.items():
        if re.search(pattern, text, re.IGNORECASE):
            replacement = random.choice(options)
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
    return text


# ── Step 3: Sentence-opening variation ───────────────────────────────────────

def vary_opening(sentence: str) -> str:
    """
    Apply structural opening variation:
    - Inversion patterns ("X is key to Y" → "Key to Y is X")
    - Subordinate clause fronting
    """
    sentence = sentence.strip()

    # Try inversion patterns first
    for pattern, builder in OPENING_INVERSIONS:
        m = re.match(pattern, sentence, re.IGNORECASE)
        if m:
            try:
                rewritten = builder(m)
                if rewritten and rewritten != sentence:
                    return rewritten
            except Exception:
                pass

    # Subordinate clause fronting (same logic as paraphraser but stricter)
    match = re.search(SUBORDINATORS, sentence, re.IGNORECASE)
    if match:
        pivot = match.start()
        if pivot > 10:  # subordinator is mid-sentence — front it
            main = sentence[:pivot].strip().rstrip(",")
            sub = sentence[pivot:].strip().rstrip(".")
            return f"{sub.capitalize()}, {main[0].lower() + main[1:]}."

    return sentence


# ── Step 4: Burstiness injection ─────────────────────────────────────────────

def inject_burstiness(sentences: list[str]) -> list[str]:
    """
    Vary sentence lengths to increase burstiness:
    - Split very long sentences (>35 words) at coordinating conjunctions.
    - Merge consecutive very short sentences (<6 words) with a connective.
    """
    # Split long sentences
    expanded: list[str] = []
    for sent in sentences:
        words = word_tokenize(sent)
        if len(words) > 35:
            # Try splitting at ", and", ", but", ", which", ", so"
            split_match = re.search(
                r",\s+(and|but|which|so|yet|while)\s+",
                sent, re.IGNORECASE
            )
            if split_match:
                left = sent[:split_match.start()].strip().rstrip(",")
                right = sent[split_match.end():].strip().rstrip(".")
                # Capitalise right half and make it a standalone sentence
                if right:
                    right = right[0].upper() + right[1:] + "."
                if left and not left.endswith("."):
                    left += "."
                expanded.extend([left, right] if right else [left])
                continue
        expanded.append(sent)

    # Merge consecutive short sentences
    merged: list[str] = []
    i = 0
    connectives = ["That said,", "In doing so,", "As a result,", "This means that"]
    while i < len(expanded):
        curr = expanded[i]
        if (
            i + 1 < len(expanded)
            and len(word_tokenize(curr)) < 6
            and len(word_tokenize(expanded[i + 1])) < 6
        ):
            conn = random.choice(connectives)
            next_sent = expanded[i + 1].rstrip(".")
            merged.append(f"{curr.rstrip('.')} — {conn.lower()} {next_sent[0].lower() + next_sent[1:]}.")
            i += 2
        else:
            merged.append(curr)
            i += 1

    return merged


# ── Step 1: Synonym swap (light — humanizer uses lower intensity by default) ──

def synonym_swap(sentence: str, intensity: float = 0.3) -> str:
    """Lightly replace eligible words with common synonyms."""
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
        synonyms = _get_synonyms(word.lower(), tag)
        if synonyms:
            result.append(_preserve_case(word, random.choice(synonyms)))
        else:
            result.append(word)
    text = " ".join(result)
    text = re.sub(r" ([,.!?;:])", r"\1", text)
    return text


# ── Format-preserving processor ──────────────────────────────────────────────

_BULLET_RE = re.compile(r'^([•\-\*]|\d+[\.\)])\s+')


def _apply_to_line(line: str, intensity: float) -> str:
    """Run the per-sentence humanization pipeline on a single line's content."""
    sentences = sent_tokenize(line)
    processed = []
    for sent in sentences:
        s = inject_hedging(sent)
        s = vary_opening(s)
        s = synonym_swap(s, intensity)
        s = re.sub(r" ([,.!?;:])", r"\1", s)
        processed.append(s)
    processed = inject_burstiness(processed)
    return " ".join(processed)


def _process_preserving_format(text: str, intensity: float) -> str:
    """
    Apply humanization while preserving:
    - Paragraph breaks  (\n\n)
    - Line breaks       (\n)
    - Leading indentation
    - Bullet / list prefixes (•, -, *, 1., 1))
    - Blank lines
    - Punctuation
    """
    result_paragraphs = []

    for para in text.split("\n\n"):
        result_lines = []
        for line in para.split("\n"):
            if not line.strip():
                result_lines.append(line)
                continue

            # Preserve leading whitespace
            indent = line[: len(line) - len(line.lstrip())]
            content = line[len(indent):]

            # Preserve bullet/list prefix
            bm = _BULLET_RE.match(content)
            prefix = bm.group(0) if bm else ""
            content = content[len(prefix):]

            if content.strip():
                content = _apply_to_line(content, intensity)

            result_lines.append(indent + prefix + content)

        result_paragraphs.append("\n".join(result_lines))

    return "\n\n".join(result_paragraphs)


# ── Main humanize function ────────────────────────────────────────────────────

def humanize(text: str, intensity: float = 0.3) -> str:
    """
    Apply the full humanization pipeline to input text, preserving all formatting.

    Pipeline:
      1. Expand contractions
      2. Remove filler phrases
      3. Per-paragraph → per-line → per-sentence: hedging, opening variation,
         synonym swap, burstiness injection
      4. Transition diversification on full text
      5. Final cleanup
    """
    # Steps 1-2 are regex-based and safe to run on the full text
    text = _expand_contractions(text)
    text = remove_filler(text)

    # Steps 3-4: format-preserving sentence-level transforms
    output = _process_preserving_format(text, intensity)
    output = diversify_transitions(output)

    output = re.sub(r"(?m)(?<=\S)  +", " ", output).strip()
    return output
