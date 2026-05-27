import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.shot_frames import frame_ranges_from_counts, nearest_4n_plus_1


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

            if "frames_per_shot" in caption_data:
                frames_per_shot = caption_data["frames_per_shot"]
                if len(frames_per_shot) != len(keys):
                    raise ValueError(
                        f"frames_per_shot length mismatch in {json_path}: "
                        f"{len(frames_per_shot)} vs {len(keys)} shots"
                    )
            else:
                frames_per_shot = [args.frames_per_shot] * len(keys)

            frames_per_shot = [
                nearest_4n_plus_1(frame_count)
                for frame_count in frames_per_shot
            ]
            frame_number = frame_ranges_from_counts(frames_per_shot)
            writer.writerow(
                {
                    "shot_num_from_caption": len(keys),
                    "json_path": json_path.as_posix(),
                    "frame_number": json.dumps(frame_number),
                }
            )


if __name__ == "__main__":
    main()
