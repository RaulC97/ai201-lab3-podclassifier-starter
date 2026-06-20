import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    task_instruction = """You are classifying podcast episodes by their format. Classify the episode into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or discussing a topic together
- narrative: a story assembled from external sources — interviews, archival audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy."""

    lines = [task_instruction, ""]

    if labeled_examples:
        lines.append("Here are some labeled examples:")
        lines.append("")
        for ex in labeled_examples:
            lines.append(f"Title: {ex['title']}")
            lines.append(f"Description: {ex['description']}")
            lines.append(f"Label: {ex['label']}")
            lines.append("---")
        lines.append("")

    lines.append("Now classify this episode:")
    lines.append("")
    lines.append(f"Description: {description}")
    lines.append("Label: ?")
    lines.append("")
    lines.append("Classify the episode above. Return your answer in exactly this format:")
    lines.append("Label: <one of interview, solo, panel, narrative>")
    lines.append("Reasoning: <one or two sentences explaining why>")

    return "\n".join(lines)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        # Step 1: Build the prompt
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2: Send to the LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        raw_text = response.choices[0].message.content or ""

        # Step 3: Parse the response
        label = "unknown"
        reasoning = ""
        for line in raw_text.strip().splitlines():
            if line.startswith("Label:"):
                label = line.split("Label:", 1)[1].strip().lower()
            elif line.startswith("Reasoning:"):
                reasoning = line.split("Reasoning:", 1)[1].strip()

        # Step 4: Validate the label
        if label not in VALID_LABELS:
            label = "unknown"

        return {"label": label, "reasoning": reasoning}

    # Step 5: Handle errors gracefully
    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error: {e}"}
