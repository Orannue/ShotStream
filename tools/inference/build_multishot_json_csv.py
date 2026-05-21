import argparse
import csv
import json
from pathlib import Path


def shot_keys(caption_data):
    keys = []
    index = 1
    while f"shot{index}" in caption_data:
        keys.append(f"shot{index}")
        index += 1
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--frames_per_shot", type=int, default=81)
    args = parser.parse_args()

    json_paths = sorted(args.json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {args.json_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["shot_num_from_caption", "json_path", "frame_number"],
        )
        writer.writeheader()

        for json_path in json_paths:
            with json_path.open("r", encoding="utf-8") as caption_file:
                caption_data = json.load(caption_file)

            keys = shot_keys(caption_data)
            if not keys:
                raise ValueError(f"No shot captions found in {json_path}")

            frame_number = [
                [i * args.frames_per_shot, (i + 1) * args.frames_per_shot]
                for i in range(len(keys))
            ]
            writer.writerow(
                {
                    "shot_num_from_caption": len(keys),
                    "json_path": json_path.as_posix(),
                    "frame_number": json.dumps(frame_number),
                }
            )


if __name__ == "__main__":
    main()
