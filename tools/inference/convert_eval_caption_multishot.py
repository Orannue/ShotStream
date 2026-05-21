import argparse
import json
import re
from pathlib import Path


SHOT_OPENERS = [
    "The camera cuts to",
    "The scene cuts to",
    "The camera cuts back to",
    "The scene cuts back to",
    "The shot returns to",
]

ACTION_VERBS = [
    "adjusting", "addressing", "balancing", "breaking", "bracing", "brushing",
    "catching", "checking", "clapping", "clasping", "cooling", "crouching",
    "doing", "dropping", "examining", "extending", "finishing", "folding",
    "gesturing", "grabbing", "holding", "hoisting", "jogging", "kneeling",
    "leaning", "lifting", "looking", "opening", "pausing", "performing", "picking",
    "pointing", "pressing", "pushing", "raising", "reaching", "reading",
    "rolling", "running", "setting", "sitting", "sliding", "smiling",
    "snapping", "speaking", "spinning", "standing", "steering", "switching",
    "taking", "tapping", "tearing", "throwing", "tucking", "typing",
    "rising", "sharing", "unfolding", "walking", "waving", "wiping",
]

NON_ACTION_ING_WORDS = {
    "wearing",
    "glowing",
    "floating",
    "lighting",
    "shopping",
    "morning",
    "evening",
    "setting",
}


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_shot_marker(prompt):
    text = re.sub(r"^\s*\[shot cut\]\s*", "", prompt).strip()
    text = re.sub(r"^\[character1\]\s*,\s*", "", text).strip()
    return normalize_spaces(text)


def replace_character_tags(text):
    replacements = {
        "[character1]": "the subject",
        "[character2]": "the second subject",
        "[character3]": "the third subject",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("[shot cut]", "")
    return normalize_spaces(text)


def infer_alias(prompts):
    joined = " ".join(prompts).lower()
    if "[character2]" in joined or "[character3]" in joined:
        return "the subjects"
    first = strip_shot_marker(prompts[0]).lower()
    if any(word in first for word in ["grandfather", "father", "businessman", " man ", " male ", "boy"]):
        return "the man"
    if any(word in first for word in ["woman", "female", "girl", "traveler", "shopper", "mother"]):
        return "the woman"
    return "the subject"


def action_start_index(text):
    matches = []
    for verb in ACTION_VERBS:
        match = re.search(rf"\b{verb}\b", text, flags=re.IGNORECASE)
        if match:
            matches.append(match.start())
    for match in re.finditer(r"\b[a-zA-Z]+ing\b", text):
        if match.group(0).lower() not in NON_ACTION_ING_WORDS:
            matches.append(match.start())
    return min(matches) if matches else None


def split_first_sentence(text):
    parts = text.split(".", 1)
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return first, rest


def replace_initial_character_description(prompt, alias):
    text = replace_character_tags(strip_shot_marker(prompt))
    first_sentence, rest = split_first_sentence(text)
    start = action_start_index(first_sentence)
    if start is None:
        # Fallback for unusual templates: only remove the leading character-description clause.
        clauses = first_sentence.split(",")
        if len(clauses) > 1:
            first_sentence = f"{alias} {','.join(clauses[1:]).strip()}"
        else:
            first_sentence = f"{alias} {first_sentence}"
    else:
        first_sentence = f"{alias} {first_sentence[start:].strip()}"

    return f"{first_sentence}. {rest}".strip() if rest else f"{first_sentence}."


def collect_character_context(prompts):
    context = []
    for prompt in prompts:
        text = replace_character_tags(strip_shot_marker(prompt))
        first_sentence, _ = split_first_sentence(text)
        start = action_start_index(first_sentence)
        if start is None:
            candidate = first_sentence
        else:
            candidate = first_sentence[:start].strip(" ,")
        if candidate and candidate not in context:
            context.append(candidate)
    return context


def build_global_caption(item, prompts):
    summary = replace_character_tags(item["random_concept_summary"])
    character_context = collect_character_context(prompts)
    if character_context:
        return (
            f"{summary} The main subject descriptions and shared visual context are: "
            f"{'; '.join(character_context)}. The video plays at normal speed."
        )
    return f"{summary} The video plays at normal speed."


def convert_item(item):
    prompts = item["prompts"]
    alias = infer_alias(prompts)
    output = {
        "idx": item["index"],
        "global_caption": build_global_caption(item, prompts),
    }

    for shot_index, prompt in enumerate(prompts, start=1):
        if shot_index == 1:
            output[f"shot{shot_index}"] = f"The video opens with {replace_character_tags(strip_shot_marker(prompt))}"
        else:
            opener = SHOT_OPENERS[(shot_index - 2) % len(SHOT_OPENERS)]
            output[f"shot{shot_index}"] = f"{opener} {replace_initial_character_description(prompt, alias)}"
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as file:
        items = json.load(file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        output = convert_item(item)
        output_path = args.output_dir / f"{item['index']:03d}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
            file.write("\n")

    print(f"Converted {len(items)} files into {args.output_dir}")


if __name__ == "__main__":
    main()
