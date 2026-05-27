import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.shot_frames import nearest_4n_plus_1


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON list in {path}")
    return data


def sample_id(item):
    return str(item.get("sample_id", item.get("idx", item.get("index"))))


def generation_items_by_id(path):
    if path is None:
        return {}
    return {sample_id(item): item for item in load_json(path)}


def target_boundaries_from_frames(frames_per_shot):
    boundaries = []
    elapsed = 0
    for frame_count in frames_per_shot[:-1]:
        elapsed += nearest_4n_plus_1(frame_count)
        boundaries.append(elapsed)
    return boundaries


def main():
    parser = argparse.ArgumentParser(
        description="Build a VBench multi-shot manifest from MSAVBench evaluation input templates."
    )
    parser.add_argument("--template_json", type=Path, required=True)
    parser.add_argument("--generation_json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result_root", default="demo/infer/generation_inputs_video_only")
    parser.add_argument("--video_id_template", default="video{sample_id}")
    parser.add_argument("--shot_file_template", default="shot{shot_id}.mp4")
    parser.add_argument("--full_video_name", default="full.mp4")
    args = parser.parse_args()

    template_items = load_json(args.template_json)
    generation_by_id = generation_items_by_id(args.generation_json)

    manifest = {}
    for index, item in enumerate(template_items):
        sid = sample_id(item)
        if sid == "None":
            sid = str(index)
        video_id = args.video_id_template.format(sample_id=sid, zero_based=index, one_based=index + 1)
        video_dir = f"{args.result_root}/{video_id}"

        shot_captions = item.get("shot_captions") or []
        if not shot_captions:
            raise ValueError(f"Template item {sid} has no shot_captions")

        generation_item = generation_by_id.get(sid, {})
        frames_per_shot = generation_item.get("frames_per_shot")
        if frames_per_shot is not None and len(frames_per_shot) != len(shot_captions):
            raise ValueError(
                f"Item {sid} frames_per_shot length {len(frames_per_shot)} "
                f"does not match shot_captions length {len(shot_captions)}"
            )

        shots = []
        for shot_id, caption in enumerate(shot_captions, start=1):
            shots.append(
                {
                    "id": shot_id,
                    "file": args.shot_file_template.format(shot_id=shot_id),
                    "caption": caption,
                    "characters": [],
                }
            )

        entry = {
            "dir": video_dir,
            "full_video": f"{video_dir}/{args.full_video_name}",
            "global_caption": item.get("prompt"),
            "shots": shots,
            "characters": {},
            "source": {
                "sample_id": sid,
                "template_video_path": item.get("video_path"),
            },
        }
        if frames_per_shot is not None:
            normalized_frames = [nearest_4n_plus_1(frame_count) for frame_count in frames_per_shot]
            entry["source"]["frames_per_shot"] = normalized_frames
            entry["target_boundaries_frames"] = target_boundaries_from_frames(normalized_frames)

        manifest[video_id] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Wrote manifest with {len(manifest)} videos to {args.output}")


if __name__ == "__main__":
    main()
