"""Wan2.1 inference with an explicit DynamicRad configuration.

Run as ``python -m scripts.inference_wan`` from the repository root.
"""

import argparse
import csv
import json
import time
from pathlib import Path


def load_profile(path, regime, mode):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["Scenario"].lower() == regime and row["Mode"] == mode]
    if len(matches) != 1:
        raise ValueError(f"Expected one {regime}/{mode} profile in {path}; found {len(matches)}")
    row = matches[0]
    profile = {
        "decay_factor": float(row["Decay"]),
        "long_factor": float(row["Long"]),
        "mask_threshold": float(row["Mask_Th"]),
        "col_density_threshold": float(row["Col_Th"]),
    }
    if mode == "static_ratio":
        profile["topk_ratio1"] = float(row["Ratio1/Near"])
        profile["topk_ratio2"] = float(row["Ratio2/Far"])
    else:
        profile["near_frame_threshold"] = float(row["Ratio1/Near"])
        profile["far_frame_threshold"] = float(row["Ratio2/Far"])
    return profile


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_file", default="outputs/wan_sample.mp4")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=69)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pattern", choices=("radial", "dense"), default="radial")
    parser.add_argument("--topk_mode", choices=("static_ratio", "dynamic_threshold"),
                        default="dynamic_threshold")
    parser.add_argument("--profile_csv", help="Offline BO CSV; motion regime is selected explicitly")
    parser.add_argument("--motion_regime", choices=("low", "mid", "high"))
    parser.add_argument("--block_size", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--decay_factor", type=float, default=1.6)
    parser.add_argument("--long_factor", type=float, default=0.9)
    parser.add_argument("--mask_threshold", type=float, default=0.55)
    parser.add_argument("--col_density_threshold", type=float, default=0.40)
    parser.add_argument("--topk_ratio1", type=float, default=0.55)
    parser.add_argument("--topk_ratio2", type=float, default=0.50)
    parser.add_argument("--topk_seed", type=int, default=0)
    parser.add_argument("--near_frame_threshold", type=float, default=-2.0)
    parser.add_argument("--far_frame_threshold", type=float, default=2.4)
    parser.add_argument("--dense_layers", type=int, default=0)
    parser.add_argument("--dense_timesteps", type=int, default=0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.profile_csv) != bool(args.motion_regime):
        parser.error("--profile_csv and --motion_regime must be supplied together")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("Wan num_frames must be 4n+1")
    if args.height % 16 or args.width % 16:
        parser.error("Wan height and width must be multiples of 16")

    import torch
    if not torch.cuda.is_available():
        parser.error("Wan2.1 inference requires a CUDA GPU")
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    from models.wan2_1.inference import replace_wan_attention
    from models.wan2_1.sparse_transformer import replace_sparse_forward

    config = {key: getattr(args, key) for key in (
        "decay_factor", "long_factor", "mask_threshold", "col_density_threshold",
        "topk_ratio1", "topk_ratio2", "near_frame_threshold", "far_frame_threshold")}
    if args.profile_csv:
        config.update(load_profile(args.profile_csv, args.motion_regime, args.topk_mode))

    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    replace_sparse_forward()
    replace_wan_attention(
        pipe, height=args.height, width=args.width, num_frames=args.num_frames,
        dense_layers=args.dense_layers, dense_timesteps=args.dense_timesteps,
        sparsity_type=args.pattern, block_size=args.block_size,
        topk_mode=args.topk_mode, topk_seed=args.topk_seed, **config,
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started = time.perf_counter()
    result = pipe(
        prompt=args.prompt, height=args.height, width=args.width,
        num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, generator=generator,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(output), fps=16)
    metadata = {
        "model_id": args.model_id, "prompt": args.prompt, "seed": args.seed,
        "pattern": args.pattern, "topk_mode": args.topk_mode,
        "motion_regime": args.motion_regime, "profile_csv": args.profile_csv,
        "config": config, "generation_seconds": elapsed,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Video: {output}\nMetadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
