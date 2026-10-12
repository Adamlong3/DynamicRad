"""Search DynamicRad mask configurations on an AR feature-drift proxy.

Run from the repository root with ``python -m bo_pipeline.run_bo_pipeline``.
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from attention.attn_mask import gen_log_mask_shrinked


DRIFT = {"low": 0.02, "mid": 0.15, "high": 0.40}


def make_proxy_features(regime, frames, tokens_per_frame, heads, head_dim, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn((1, tokens_per_frame, heads, head_dim),
                        generator=generator, device=device)
    features = []
    for _ in range(frames):
        noise = torch.randn(state.shape, generator=generator, device=device)
        state = (1 - DRIFT[regime]) * state + DRIFT[regime] * noise
        features.append(F.normalize(state, dim=-1) * math.sqrt(head_dim))
    q = torch.cat(features, dim=1)
    return q, q.clone()


def dense_attention_weights(q, k):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)


def reconstruction_error(dense_weights, block_mask, block_size):
    token_mask = block_mask.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    # All intra-frame blocks stay dense, so every query has a valid key.
    sparse_weights = dense_weights.masked_fill(~token_mask, 0)
    sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return ((dense_weights - sparse_weights).square().sum()
            / dense_weights.square().sum().clamp_min(1e-12))


def evaluate_trial(trial, q, k, dense_weights, frames, block_size, mode,
                   target_sparsity, seed):
    mask_threshold = trial.suggest_float("mask_threshold", 0.1, 1.0)
    col_density_threshold = trial.suggest_float("col_density_threshold", 0.1, 1.0)
    decay_factor = trial.suggest_float("decay_factor", 1.0, 3.0)
    long_factor = trial.suggest_float("long_factor", 0.1, 1.0)

    params = {}
    if mode == "static_ratio":
        params["topk_ratio1"] = trial.suggest_float("topk_ratio1", 0.3, 1.0)
        params["topk_ratio2"] = trial.suggest_float("topk_ratio2", 0.1, 0.8)
    else:
        params["near_frame_threshold"] = trial.suggest_float("near_frame_threshold", -10.0, 5.0)
        params["far_frame_threshold"] = trial.suggest_float("far_frame_threshold", -5.0, 8.0)

    mask = gen_log_mask_shrinked(
        q, k, s=q.shape[1], video_token_num=q.shape[1], num_frame=frames,
        block_size=block_size, model_type="wan", topk_mode=mode,
        decay_factor=decay_factor, long_factor=long_factor,
        mask_threshold=mask_threshold, col_density_threshold=col_density_threshold,
        topk_seed=seed, **params,
    )
    sparsity = 1.0 - mask.float().mean().item()
    error = reconstruction_error(dense_weights, mask, block_size).item()
    trial.set_user_attr("sparsity", sparsity)
    trial.set_user_attr("normalized_mse", error)
    return error + 10.0 * max(0.0, target_sparsity - sparsity)


def run_search(args):
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Install requirements.txt to run the Optuna BO example") from exc

    if args.frames < 2 or args.tokens_per_frame % args.block_size:
        raise SystemExit("Use at least two frames and a tokens-per-frame multiple of block-size")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for regime_idx, regime in enumerate(DRIFT):
        q, k = make_proxy_features(regime, args.frames, args.tokens_per_frame,
                                   args.heads, args.head_dim, device, args.seed + regime_idx)
        dense_weights = dense_attention_weights(q, k)
        for mode_idx, mode in enumerate(("static_ratio", "dynamic_threshold")):
            sampler = optuna.samplers.TPESampler(seed=args.seed + regime_idx * 2 + mode_idx)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda trial: evaluate_trial(trial, q, k, dense_weights, args.frames,
                                             args.block_size, mode, args.target_sparsity,
                                             args.seed),
                n_trials=args.steps,
            )
            best = study.best_trial
            records.append({
                "Scenario": regime,
                "Mode": mode,
                "Decay": best.params["decay_factor"],
                "Long": best.params["long_factor"],
                "Mask_Th": best.params["mask_threshold"],
                "Col_Th": best.params["col_density_threshold"],
                "Ratio1/Near": best.params.get("topk_ratio1", best.params.get("near_frame_threshold")),
                "Ratio2/Far": best.params.get("topk_ratio2", best.params.get("far_frame_threshold")),
                "Best_Sparsity": best.user_attrs["sparsity"],
                "Best_Loss": best.value,
            })
            print(f"{regime:>4} {mode:<17} sparsity={best.user_attrs['sparsity']:.1%} "
                  f"loss={best.value:.4f}")

    path = output_dir / f"final_bo_lookup_table_steps{args.steps}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"AR-proxy lookup table: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--tokens-per-frame", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32, choices=(32, 64, 128))
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--target-sparsity", type=float, default=0.8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/bo")
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.target_sparsity <= 1:
        parser.error("--steps must be positive and --target-sparsity must be in [0, 1]")
    run_search(args)


if __name__ == "__main__":
    main()
