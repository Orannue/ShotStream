import argparse
import ast
import json
import logging
import os
import re
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalInferenceArPipeline
from utils.dataset import MultiShots_FrameConcat_Dataset
from utils.misc import set_seed
from utils.shot_frames import frame_ranges_from_counts, nearest_4n_plus_1


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def read_json_idx(json_path, fallback_idx):
    with open(json_path, "r", encoding="utf-8") as file:
        caption_data = json.load(file)

    json_idx = caption_data.get("idx", caption_data.get("index", caption_data.get("sample_id")))
    if json_idx is None:
        match = re.search(r"\d+", os.path.basename(json_path))
        json_idx = match.group(0) if match else fallback_idx
    return json_idx


def as_int_list(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(as_int_list(item))
        return output
    return [int(value)]


def clipped_frame_ranges(total_frames, frame_counts):
    ranges = []
    for start_frame, end_frame in frame_ranges_from_counts(frame_counts):
        ranges.append([start_frame, min(end_frame, total_frames)])
    return ranges


def expected_video_files(video_dir, shot_count):
    return [os.path.join(video_dir, "full.mp4")] + [
        os.path.join(video_dir, f"shot{shot_idx}.mp4")
        for shot_idx in range(1, shot_count + 1)
    ]


def output_complete(video_dir, shot_count):
    return all(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in expected_video_files(video_dir, shot_count)
    )


def replace_output_dir(tmp_video_dir, video_dir):
    if os.path.isdir(video_dir):
        shutil.rmtree(video_dir)
    os.replace(tmp_video_dir, video_dir)


def shot_count_for_row(dataset, row_idx):
    if dataset.frame_number is not None:
        return len(ast.literal_eval(dataset.frame_number[row_idx]))

    with open(dataset.caption_json_path[row_idx], "r", encoding="utf-8") as file:
        caption_data = json.load(file)
    return len([key for key in caption_data if re.fullmatch(r"shot\d+", key)])


def summarize_outputs(dataset, output_folder):
    complete = []
    incomplete = []
    missing = []

    for row_idx, json_path in enumerate(dataset.caption_json_path):
        json_idx = read_json_idx(json_path, row_idx)
        shot_count = shot_count_for_row(dataset, row_idx)
        video_dir = os.path.join(output_folder, f"video{json_idx}")

        if not os.path.isdir(video_dir):
            missing.append(json_idx)
        elif output_complete(video_dir, shot_count):
            complete.append(json_idx)
        else:
            incomplete.append(json_idx)

    return complete, incomplete, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--default_config_path", type=str, default="ckpts/default_config.yaml")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--resume_lora_ckpt", type=str, default=None)
    parser.add_argument("--model_root", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--frames_per_shot", type=int, default=81)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--multi_caption", type=str_to_bool, default=True)
    parser.add_argument("--use_wo_rope_cache", type=str_to_bool, default=False)
    args = parser.parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    print(f"Device is {device}")
    _init_logging(rank)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    logging.info(f"Generation job args: {args}")

    if dist.is_initialized():
        base_seed = [args.seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.seed = base_seed[0]

    current_seed = args.seed + rank
    set_seed(current_seed)
    logging.info(f"Rank {rank} set seed to {current_seed}")

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(args.default_config_path)
    config = OmegaConf.merge(default_config, config)

    config.multi_caption = args.multi_caption
    config.use_wo_rope_cache = args.use_wo_rope_cache
    if args.model_root is not None:
        config.model_root = args.model_root

    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferenceArPipeline(config, device=device)
    else:
        raise ValueError("Config must define denoising_step_list")

    if args.resume_ckpt is not None:
        config.resume_ckpt = args.resume_ckpt

    if config.resume_ckpt:
        state_dict = torch.load(config.resume_ckpt, map_location="cpu")
        print(f"resume generator's ckpt from {config.resume_ckpt}")
        pipeline.generator.load_state_dict(state_dict["generator"])

    pipeline = pipeline.to(dtype=torch.bfloat16)

    if args.data_path is not None:
        config.data_path = args.data_path
    print(f"Dataset Path is {config.data_path}")

    dataset = MultiShots_FrameConcat_Dataset(csv_path=config.data_path)
    num_prompts = len(dataset)
    print(f"Number of prompts: {num_prompts}")

    if dist.is_initialized():
        rank_indices = list(range(rank, len(dataset), world_size))
        dataset_for_loader = Subset(dataset, rank_indices)
        sampler = SequentialSampler(dataset_for_loader)
        logging.info(
            "Rank %d/%d processing %d samples",
            rank,
            world_size,
            len(rank_indices),
        )
    else:
        dataset_for_loader = dataset
        sampler = SequentialSampler(dataset_for_loader)
    dataloader = DataLoader(dataset_for_loader, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    for _, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        row_idx = batch_data["idx"].item()
        batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

        json_path = dataset.caption_json_path[row_idx]
        json_idx = read_json_idx(json_path, row_idx)
        shot_count = len(batch["shots_captions"][0])
        video_dir = os.path.join(args.output_folder, f"video{json_idx}")

        if output_complete(video_dir, shot_count):
            logging.info("Skip video%s: existing output is complete", json_idx)
            continue

        video = pipeline.inference(
            batch=batch,
            use_wo_rope_cache=config.use_wo_rope_cache,
        )
        video = rearrange(video, "b t c h w -> b t h w c").cpu()
        video = 255.0 * video

        pipeline.vae.model.clear_cache()

        frame_counts = as_int_list(batch.get("shot_frame_counts"))
        if not frame_counts:
            frame_counts = [args.frames_per_shot] * shot_count
        frame_counts = [nearest_4n_plus_1(frame_count) for frame_count in frame_counts]
        if len(frame_counts) != shot_count:
            raise ValueError(
                "frames_per_shot length must match shot count for "
                f"video{json_idx}: {len(frame_counts)} vs {shot_count}"
            )
        frame_ranges = clipped_frame_ranges(video.shape[1], frame_counts)

        tmp_video_dir = tempfile.mkdtemp(
            dir=args.output_folder,
            prefix=f".video{json_idx}.rank{rank}.tmp-",
        )
        try:
            write_video(os.path.join(tmp_video_dir, "full.mp4"), video[0], fps=16)
            for shot_idx, (start_frame, end_frame) in enumerate(frame_ranges, start=1):
                if start_frame >= end_frame:
                    logging.warning(
                        "Skip empty shot%d for video%s: range [%d, %d) exceeds full video length %d",
                        shot_idx,
                        json_idx,
                        start_frame,
                        end_frame,
                        video.shape[1],
                    )
                    continue
                write_video(
                    os.path.join(tmp_video_dir, f"shot{shot_idx}.mp4"),
                    video[0, start_frame:end_frame],
                    fps=16,
                )

            if not output_complete(tmp_video_dir, shot_count):
                raise RuntimeError(f"Incomplete output for video{json_idx} in {tmp_video_dir}")

            with open(os.path.join(tmp_video_dir, ".done"), "w", encoding="utf-8") as done_file:
                json.dump(
                    {
                        "row_idx": row_idx,
                        "json_idx": json_idx,
                        "shot_count": shot_count,
                        "frames": int(video.shape[1]),
                        "frames_per_shot": frame_counts,
                    },
                    done_file,
                    ensure_ascii=False,
                    indent=2,
                )
                done_file.write("\n")

            replace_output_dir(tmp_video_dir, video_dir)
            tmp_video_dir = None
            logging.info("Wrote complete output for video%s", json_idx)
        finally:
            if tmp_video_dir is not None and os.path.isdir(tmp_video_dir):
                shutil.rmtree(tmp_video_dir)

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        complete, incomplete, missing = summarize_outputs(dataset, args.output_folder)
        logging.info(
            "Output summary: %d/%d complete, %d incomplete, %d missing",
            len(complete),
            len(dataset),
            len(incomplete),
            len(missing),
        )
        if incomplete or missing:
            raise RuntimeError(
                "Generation incomplete. "
                f"Incomplete videos: {incomplete}. Missing videos: {missing}."
            )


if __name__ == "__main__":
    main()
