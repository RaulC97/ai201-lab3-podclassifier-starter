# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Request this exact format:

Label: <one of interview, solo, panel, narrative>
Reasoning: <one or two sentences explaining why>

Why this format?
- "Label: X" on its own line is trivial to parse: split on "Label:", take
  the first token, strip whitespace, lowercase.
- "Reasoning: Y" captures the explanation without free-form placement.
- Avoids JSON (fragile — LLMs sometimes add prose around the braces) and
  avoids a bare label (gives no reasoning to surface to the user).
- Both fields appear on predictable lines, so you don't need a regex —
  a simple split + strip is enough.
```

---

**Edge cases to handle in the prompt:**

```
1. labeled_examples is empty:
   The examples section should simply be omitted (or replaced with a note
   like "No examples available."). The task instruction and output format
   are still present, so the LLM can still attempt a zero-shot classification.
   Don't crash — just skip the loop that adds example blocks.

2. Description is very short (e.g., a single sentence or just a title):
   Include whatever is there. Don't pad or modify it. The LLM handles terse
   descriptions; forcing extra text would misrepresent the input.

3. Description contains special characters or newlines:
   No escaping needed — this is plain text sent as a string, not JSON.
   Just include it verbatim between the "Description:" line and the next block.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The LLM response will look like:
  Label: interview
  Reasoning: The description mentions a host interviewing a guest expert.

Parsing steps:
  1. Split the response text by newline: lines = raw_text.strip().splitlines()
  2. Find the line that starts with "Label:" — extract everything after the colon,
     strip whitespace, and lowercase it:
       label = line.split("Label:", 1)[1].strip().lower()
  3. Find the line that starts with "Reasoning:" — extract everything after the
     colon and strip whitespace:
       reasoning = line.split("Reasoning:", 1)[1].strip()
  4. If either line is not found, set the corresponding value to "" or "unknown".

You can also do this with a simple loop over lines rather than searching the
whole text at once — either approach works.
```

---

**Step 4 — Validate the label:**

```
After parsing, check:
  if label not in VALID_LABELS:
      label = "unknown"

VALID_LABELS = ["interview", "solo", "panel", "narrative"]

This handles cases where the LLM returns something like "Interview" (wrong case —
already handled by .lower()), "N/A", a full sentence instead of a single word,
or an empty string. Setting it to "unknown" keeps the return shape consistent
and lets the evaluation loop count how many episodes couldn't be classified.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the entire function body in a try/except block:

  try:
      # steps 1-4 here
      return {"label": label, "reasoning": reasoning}
  except Exception as e:
      return {"label": "unknown", "reasoning": f"Error: {e}"}

Things that can go wrong:
  - Network/API error (Groq is unreachable, rate limit hit, bad API key)
  - The LLM returns an empty response or None
  - The response doesn't contain "Label:" or "Reasoning:" at all (unparseable)
  - The response has extra formatting (markdown bold, bullet points, etc.)

In all cases, returning {"label": "unknown", "reasoning": "Error: ..."} keeps
the evaluation loop running across all 20 episodes instead of crashing partway.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [title]
Raw response text: [paste it here]
```

**How did you parse the label out of the response?**

```
[describe the string operations — strip, split, lower, etc.]
```

**Did any episodes return `"unknown"`? If so, why?**

```
[yes / no — if yes, what did the raw response look like?]
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
