import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.shot_frames import nearest_4n_plus_1


GLOBAL_CAPTION_OVERRIDES = {
    "21": (
        "The video contains 6 shots. It is a cyberpunk-style action short set inside "
        "a medieval magic castle, where an elderly woman performs a high-speed swimming "
        "training sequence in a rune-lined pool. The visual tone is low-saturation and "
        "cool blue-purple, with neon lane lines, wet stone, steam, rotating copper pipes, "
        "and cold side light emphasizing water splashes, breathing, wrinkles, and controlled "
        "athletic effort. The overall mood is intense and tense."
    ),
    "22": (
        "The video contains 10 shots. It is a realistic action training short set in a "
        "mountain-top forest camp that opens at dawn and gradually moves toward a seaside "
        "rock platform. A boy progresses from balance-beam warmup and precise footwork into "
        "a backflip, ridge running, uneven-bar swings, aerial transitions, and a firm final "
        "landing. The neutral color palette is shaped by morning natural light and later "
        "seaside backlight, with a focused emotional arc from tension to passionate intensity."
    ),
    "150": (
        "The video consists of 11 shots and is a realistic-style short film depicting a "
        "morning mountain forest rock solo performance. The overall visual tone features "
        "neutral colors with slightly cool morning mist layers, primarily using natural "
        "morning light with minimal backlighting to outline contours."
    ),
}


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}")
    return data


def read_jsonl(path):
    items = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    return items


def item_id(item):
    return str(item.get("idx", item.get("sample_id")))


def metadata_by_id(items):
    return {item_id(item): item for item in items}


def convert_item(item, metadata):
    prompts = item.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Item {item_id(item)} has no prompts list")

    frames_per_shot = item.get("frames_per_shot")
    if not isinstance(frames_per_shot, list):
        raise ValueError(f"Item {item_id(item)} has no frames_per_shot list")
    if len(frames_per_shot) != len(prompts):
        raise ValueError(
            f"Item {item_id(item)} frames_per_shot length {len(frames_per_shot)} "
            f"does not match prompts length {len(prompts)}"
        )

    source_metadata = metadata.get(item_id(item), {})
    global_caption = (
        GLOBAL_CAPTION_OVERRIDES.get(item_id(item))
        or source_metadata.get("pe_caption_overall")
    )
    if not global_caption:
        raise ValueError(f"Item {item_id(item)} has no pe_caption_overall")

    output = {
        "sample_id": item_id(item),
        "global_caption": global_caption,
        "frames_per_shot": [
            nearest_4n_plus_1(frame_count)
            for frame_count in frames_per_shot
        ],
    }

    for shot_index, prompt in enumerate(prompts, start=1):
        output[f"shot{shot_index}"] = prompt

    return output


def write_item_files(items, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        output_path = output_dir / f"{int(item['sample_id']):04d}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(item, file, ensure_ascii=False, indent=2)
            file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_generation_json", type=Path, required=True)
    parser.add_argument("--metadata_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_json_dir", type=Path, default=None)
    args = parser.parse_args()

    generation_items = read_json(args.input_generation_json)
    metadata = metadata_by_id(read_jsonl(args.metadata_jsonl))
    converted_items = [convert_item(item, metadata) for item in generation_items]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(converted_items, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if args.output_json_dir is not None:
        write_item_files(converted_items, args.output_json_dir)

    print(
        f"Converted {len(converted_items)} items into {args.output_json}"
        + (f" and {args.output_json_dir}" if args.output_json_dir else "")
    )


if __name__ == "__main__":
    main()
