import os
import sys
import argparse
from datetime import datetime

# ==============================================================================
# [CRITICAL FIX] Prevent CUDA Error: unsupported GNU version
# Forces NVCC to ignore overly high GCC versions (e.g., GCC 14/15)
# Must be executed before importing torch
# ==============================================================================
os.environ["NVCC_PREPEND_FLAGS"] = "-allow-unsupported-compiler " + os.environ.get("NVCC_PREPEND_FLAGS", "")

import torch
import torch.nn.functional as F
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import core attention modules
try:
    from dynamicrad.attention.attn_mask import RadialAttention, MaskMap
except ImportError:
    print("[Error] Cannot find attn_mask.py. Ensure you are running from the project root.")
    sys.exit(1)

# ================= 1. Fixed Configuration =================
NUM_FRAMES = 69           # Typical frames for Wan2.1-14B
HEIGHT = 768              
WIDTH = 1280              
PATCH_SIZE = 16           
TOKENS_PER_FRAME = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
BLOCK_SIZE = 32           

BATCH_SIZE = 1
NUM_HEADS = 12            
HEAD_DIM = 128            
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hard constraint for sparsity target
TARGET_SPARSITY = 0.80    

class FeatureSimulator:
    """
    Generates simulated Q/K/V features using a Drift Rate to model various motion intensities.
    (Proxy Task formulation as defined in the paper)
    """
    @staticmethod
    def generate(motion_type):
        dtype = torch.float16
        
        # Inter-frame drift rates
        if motion_type == "low":
            drift = 0.02   # Nearly static
        elif motion_type == "mid":
            drift = 0.15   # Normal motion
        elif motion_type == "high":
            drift = 0.40   # Intense motion
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")

        # Initialize the first frame (normalized to simulate LayerNorm output)
        current_frame = torch.randn(BATCH_SIZE, TOKENS_PER_FRAME, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
        current_frame = F.normalize(current_frame, dim=-1) * np.sqrt(HEAD_DIM)
        
        frames_q = []
        
        # Generate sequence via auto-regressive random walk
        for _ in range(NUM_FRAMES):
            noise = torch.randn_like(current_frame)
            current_frame = current_frame * (1 - drift) + noise * drift
            frame_normalized = F.normalize(current_frame, dim=-1) * np.sqrt(HEAD_DIM)
            frames_q.append(frame_normalized)
            
        q = torch.cat(frames_q, dim=1)
        k = q.clone()
        v = q.clone()
        
        return q, k, v

def objective(trial, motion_type, mode):
    # 1. Generate Proxy Data
    q, k, v = FeatureSimulator.generate(motion_type)
    total_tokens = q.shape[1]
    mask_map = MaskMap(video_token_num=total_tokens, num_frame=NUM_FRAMES)
    
    # 2. Compute Dense Baseline (Ground Truth)
    with torch.no_grad():
        dense_out = RadialAttention(
            query=q, key=k, value=v,
            mask_map=mask_map,
            sparsity_type="dense",
            block_size=BLOCK_SIZE,
            model_type="wan"
        )
    
    # 3. Define Search Space
    mask_threshold = trial.suggest_float("mask_threshold", 0.4, 0.8, step=0.05)
    col_density_threshold = trial.suggest_float("col_density_threshold", 0.1, 0.5, step=0.05)
    decay_factor = trial.suggest_float("decay_factor", 0.5, 3.0, step=0.1)
    long_factor = trial.suggest_float("long_factor", 0.0, 1.0, step=0.1)

    topk_ratio1 = 0.0
    topk_ratio2 = 0.0
    near_thresh = 0.0
    far_thresh = 0.0
    
    if mode == "static_ratio":
        topk_ratio1 = trial.suggest_float("topk_ratio1", 0.2, 0.8, step=0.05)
        topk_ratio2 = trial.suggest_float("topk_ratio2", 0.1, 0.6, step=0.05)
    elif mode == "dynamic_threshold":
        # FP16 DotProduct score range roughly [-5, 5]
        near_thresh = trial.suggest_float("near_frame_threshold", -2.0, 2.0, step=0.2)
        far_thresh = trial.suggest_float("far_frame_threshold", 0.0, 4.0, step=0.2)
        
    # 4. Run Sparse Attention
    try:
        with torch.no_grad():
            sparse_out = RadialAttention(
                query=q, key=k, value=v, mask_map=mask_map,
                sparsity_type="radial", block_size=BLOCK_SIZE,
                decay_factor=decay_factor, long_factor=long_factor,
                mask_threshold=mask_threshold, col_density_threshold=col_density_threshold,
                model_type="wan", topk_mode=mode,
                topk_ratio1=topk_ratio1, topk_ratio2=topk_ratio2,
                near_frame_threshold=near_thresh, far_frame_threshold=far_thresh,
                vis_frame_pair_limit=0 
            )
            
        # 5. Compute Loss and Sparsity Penalty
        generated_mask = mask_map._log_mask
        if generated_mask is None: return float('inf')
        
        real_sparsity = 1.0 - (generated_mask.sum().float() / generated_mask.numel()).item()
        
        # Normalized MSE Loss
        mse_loss = F.mse_loss(sparse_out, dense_out) / (dense_out.var() + 1e-6)
        
        # Objective = MSE * 1000 + Sparsity Penalty
        score = mse_loss.item() * 1000.0
        
        if real_sparsity < TARGET_SPARSITY:
            score += (TARGET_SPARSITY - real_sparsity) * 10000.0
            
        trial.set_user_attr("sparsity", real_sparsity)
        trial.set_user_attr("mse", mse_loss.item())
        
        return score

    except Exception:
        return float('inf')

def run_full_experiment(n_trials):
    scenarios = ["low", "mid", "high"]
    modes = ["static_ratio", "dynamic_threshold"]
    file_suffix = str(n_trials)
    
    print(f"=== Starting BO Pipeline: Steps={n_trials}, Suffix='_steps{file_suffix}' ===")
    final_report = []

    for scene in scenarios:
        for mode in modes:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> Profiling: Scenario={scene.upper()} | Mode={mode} <<<")
            
            # Suppress excessive optuna logs
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, scene, mode), n_trials=n_trials)
            
            best = study.best_params
            best_val = study.best_value
            best_attrs = study.best_trial.user_attrs
            
            print(f"  --> Best Loss: {best_val:.4f}")
            print(f"  --> Achieved Sparsity: {best_attrs['sparsity']:.2%}")
            
            df = study.trials_dataframe()
            filename = f"bo_{scene}_{mode}_steps{file_suffix}.csv"
            df.to_csv(filename)
            
            record = {
                "Scenario": scene,
                "Mode": mode,
                "Decay": round(best.get('decay_factor'), 3),
                "Long": round(best.get('long_factor'), 3),
                "Mask_Th": round(best.get('mask_threshold'), 3),
                "Col_Th": round(best.get('col_density_threshold'), 3),
                "Ratio1/Near": round(best.get('topk_ratio1') if mode=='static_ratio' else best.get('near_frame_threshold'), 3),
                "Ratio2/Far": round(best.get('topk_ratio2') if mode=='static_ratio' else best.get('far_frame_threshold'), 3),
                "Best_Sparsity": round(best_attrs['sparsity'], 5),
                "Best_Loss": round(best_val, 5)
            }
            final_report.append(record)

    df_report = pd.DataFrame(final_report)
    print("\n\n" + "="*90)
    print(f"FINAL LOOKUP TABLE (Steps={n_trials})")
    print("="*90)
    print(df_report.to_string(index=False, float_format=lambda x: "{:.5f}".format(x)))
    
    lookup_file = f"final_bo_lookup_table_steps{file_suffix}.csv"
    df_report.to_csv(lookup_file, index=False)
    print(f"\n[Info] Lookup table saved to {lookup_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Offline BO for DynamicRad")
    parser.add_argument("--steps", type=int, default=30, help="Number of trials per scenario (default: 30)")
    args = parser.parse_args()
    
    run_full_experiment(n_trials=args.steps)