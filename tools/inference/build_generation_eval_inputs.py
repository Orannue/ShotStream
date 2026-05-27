import argparse
import json
from pathlib import Path


def load_template(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Copy MSAVBench evaluation inputs while optionally pointing video_path to ShotStream outputs."
    )
    parser.add_argument("--template_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument(
        "--video_path_template",
        default="demo/infer/generation_inputs_video_only/video{sample_id}/full.mp4",
        help=(
            "Template used to rewrite video_path. Supported fields: "
            "{sample_id}, {zero_based}, {one_based}. Use empty string to keep template video_path."
        ),
    )
    args = parser.parse_args()

    items = load_template(args.template_json)
    output_items = []
    for zero_based, item in enumerate(items):
        sample_id = str(item.get("sample_id", zero_based))
        output_item = dict(item)
        if args.video_path_template:
            output_item["video_path"] = args.video_path_template.format(
                sample_id=sample_id,
                zero_based=zero_based,
                one_based=zero_based + 1,
            )
        output_items.append(output_item)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(output_items, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Wrote {len(output_items)} evaluation items to {args.output_json}")


if __name__ == "__main__":
    main()
