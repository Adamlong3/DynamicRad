#!/bin/bash
# ==============================================================================
# DynamicRad + VBench End-to-End Pipeline Script
# Functions: 1. Video Generation 2. Mask Visualization 3. VBench Eval 4. Merge Results
# ==============================================================================

# ============================== User Configurations ==============================
# 1. Generation Parameters
DENSE_LAYERS=1
DENSE_TIMESTEPS=12
DECAY_FACTOR=0.2
LONG_FACTOR=2.0
PATTERN="radial"             # "radial" for DynamicRad, "dense" for Baseline
SEED=0
NUM_FRAMES=69
HEIGHT=768
WIDTH=1280
NUM_INFERENCE_STEPS=50
BLOCK_SIZE=32
MASK_THRESHOLD=0.8
COL_DENSITY_THRESHOLD=0.5

# --- Dynamic Mode Parameters ---
TOPK_MODE="dynamic_threshold"  # Choices: static_ratio, dynamic_ratio, dynamic_threshold
TOPK_RATIO1=1.0
TOPK_RATIO2=1.0
TOPK_SEED=0
NEAR_FRAME_THRESHOLD=3.0       # Used in dynamic_threshold mode
FAR_FRAME_THRESHOLD=4.0        # Used in dynamic_threshold mode

MODEL_ID="Wan-AI/Wan2.1-T2V-14B-Diffusers" # HuggingFace Model ID or local path
PROMPT="FPV drone shot flying through a futuristic sci-fi tunnel at high speed. Repeating neon ring lights, metallic walls, motion blur, symmetrical composition, cinematic lighting, 4k."

# 2. Path Configurations (Relative to project root)
BASE_DIR="$(pwd)"
VIDEO_OUTPUT_DIR="${BASE_DIR}/outputs/experiment"
EVAL_OUTPUT_DIR="${VIDEO_OUTPUT_DIR}/evaluate"
VBENCH_CACHE_DIR="${BASE_DIR}/vbench_cache"

# Script Paths
RADIAL_SCRIPT_PATH="${BASE_DIR}/scripts/inference_wan.py"
VBENCH_EVAL_SCRIPT="${BASE_DIR}/VBench/evaluate.sh"
MERGE_SCRIPT_PATH="${BASE_DIR}/VBench/merge_results.py"
MASK_COMPARE_SCRIPT="${BASE_DIR}/scripts/compare_block_masks.py"

# Baseline dense parameters path for speedup calculation (Only effective if PATTERN=radial)
# Update this path if you have a pre-computed dense run.
DENSE_PARAMS_PATH="${BASE_DIR}/outputs/dense_baseline_params.json"

# 3. Unified Naming Convention
VIDEO_BASENAME="t3-${DENSE_LAYERS}-${DENSE_TIMESTEPS}-${NUM_FRAMES}-${BLOCK_SIZE}-${DECAY_FACTOR}-${LONG_FACTOR}-${COL_DENSITY_THRESHOLD}-${MASK_THRESHOLD}-${NUM_INFERENCE_STEPS}-${PATTERN}-${TOPK_RATIO1}-${TOPK_RATIO2}-${NEAR_FRAME_THRESHOLD}-${FAR_FRAME_THRESHOLD}-${TOPK_MODE}"
VIDEO_OUTPUT_PATH="${VIDEO_OUTPUT_DIR}/${VIDEO_BASENAME}.mp4"
RADIAL_PARAMS_FILE="${VIDEO_OUTPUT_DIR}/${VIDEO_BASENAME}_dynamicrad_params.json"

# 4. Timestamp
TIMESTAMP=$(date +"%Y-%m-%d-%H:%M:%S")

# ==============================================================================

# ============================== Step 0: Pre-process Arguments =================
PYTHON_CMD=("python" "${RADIAL_SCRIPT_PATH}")

PYTHON_CMD+=(
    "--model_id" "${MODEL_ID}"
    "--prompt" "${PROMPT}"
    "--height" "${HEIGHT}"
    "--width" "${WIDTH}"
    "--num_frames" "${NUM_FRAMES}"
    "--output_file" "${VIDEO_OUTPUT_PATH}"
    "--seed" "${SEED}"
    "--pattern" "${PATTERN}"
    "--dense_layers" "${DENSE_LAYERS}"
    "--dense_timesteps" "${DENSE_TIMESTEPS}"
    "--block_size" "${BLOCK_SIZE}"
    "--decay_factor" "${DECAY_FACTOR}"
    "--col_density_threshold" "${COL_DENSITY_THRESHOLD}"
    "--mask_threshold" "${MASK_THRESHOLD}"
    "--long_factor" "${LONG_FACTOR}"
    "--num_inference_steps" "${NUM_INFERENCE_STEPS}"
    "--guidance_scale" "5.0"
    "--topk_ratio1" "${TOPK_RATIO1}"
    "--topk_ratio2" "${TOPK_RATIO2}"
    "--topk_seed" "${TOPK_SEED}"
    "--topk_mode" "${TOPK_MODE}"
    "--near_frame_threshold" "${NEAR_FRAME_THRESHOLD}"
    "--far_frame_threshold" "${FAR_FRAME_THRESHOLD}"
)

if [ "${PATTERN}" = "radial" ]; then
    if [ -n "${DENSE_PARAMS_PATH}" ] && [ -f "${DENSE_PARAMS_PATH}" ]; then
        echo -e "\n[Pre-process] Using specified dense baseline parameters: ${DENSE_PARAMS_PATH}"
        PYTHON_CMD+=( "--dense_params_path" "${DENSE_PARAMS_PATH}" )
    else
        echo -e "\n[Pre-process] Warning: DENSE_PARAMS_PATH is not set or file does not exist."
        echo "Running without baseline mode. Speedup ratios and frame-to-frame metrics will NOT be calculated."
        read -p "Do you want to continue? (y/n): " confirm
        if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
            exit 1
        fi
    fi
fi

# ============================== Step 1: Video Generation ==============================
TOTAL_STEPS=$( [ "$PATTERN" = "radial" ] && echo "4" || echo "3" )

echo -e "\n[Step 1/${TOTAL_STEPS}] Generating video (Pattern: ${PATTERN}, Top-k Mode: ${TOPK_MODE})..."
echo "Output Path: ${VIDEO_OUTPUT_PATH}"
"${PYTHON_CMD[@]}"

if [ ! -f "${VIDEO_OUTPUT_PATH}" ]; then
    echo "[Error] Video generation failed. File not found: ${VIDEO_OUTPUT_PATH}"
    exit 1
fi
echo "[Step 1/${TOTAL_STEPS}] Video generated successfully!"

# ============================== Step 2: Mask Visualization (Radial Only) ==============
if [ "${PATTERN}" = "radial" ]; then
    echo -e "\n[Step 2/4] Generating Mask Visualization Heatmap..."
    
    # Note: Ensure Python formatting matches this bash string if debugging missing files
    MASK_FILENAME="bs${BLOCK_SIZE}_df${DECAY_FACTOR}_lf${LONG_FACTOR}_mt${MASK_THRESHOLD}_ct${COL_DENSITY_THRESHOLD}_${TOPK_RATIO1}_${TOPK_RATIO2}_${NEAR_FRAME_THRESHOLD}_${FAR_FRAME_THRESHOLD}_${TOPK_MODE}_mask.pt"
    MASK_FILE_PATH="${BASE_DIR}/visualizations/${MASK_FILENAME}"
    MASK_VIS_DIR="${VIDEO_OUTPUT_DIR}/mask_vis"
    MASK_VIS_PREFIX="${MASK_VIS_DIR}/${VIDEO_BASENAME}"
    ORIGIN_VIS_FILE="${MASK_VIS_DIR}/final_block_mask_heatmap.png"
    TARGET_VIS_FILE="${MASK_VIS_PREFIX}_heatmap.png"

    mkdir -p ${MASK_VIS_DIR}

    if [ ! -f "${MASK_FILE_PATH}" ]; then
        echo "[Error] Mask file not found: ${MASK_FILE_PATH}. Visualization skipped."
        echo "Proceeding to evaluation..."
    else
        echo -e "\nCalling ${MASK_COMPARE_SCRIPT} to generate heatmap..."
        python ${MASK_COMPARE_SCRIPT} \
            ${MASK_FILE_PATH} \
            --out_dir ${MASK_VIS_DIR}

        if [ -f "${ORIGIN_VIS_FILE}" ]; then
            mv -f ${ORIGIN_VIS_FILE} ${TARGET_VIS_FILE}
            echo "[Success] Visualization renamed to: ${TARGET_VIS_FILE}"
        else
            echo "[Warning] Default visualization file ${ORIGIN_VIS_FILE} not generated. Skipping rename."
        fi
        echo "[Step 2/4] Mask visualization completed!"
    fi
fi

# ============================== Step 3: VBench Evaluation ==============================
CURRENT_STEP=$( [ "$PATTERN" = "radial" ] && echo "3" || echo "2" )
echo -e "\n[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Evaluating video using VBench..."

if [ -f "${VBENCH_EVAL_SCRIPT}" ]; then
    bash ${VBENCH_EVAL_SCRIPT} \
        "${VIDEO_OUTPUT_PATH}" \
        "${PROMPT}" \
        "${EVAL_OUTPUT_DIR}" \
        "${VBENCH_CACHE_DIR}"

    echo -e "\nSearching for VBench result file..."
    VBENCH_RESULT_FILE=$(find ${EVAL_OUTPUT_DIR} -type f -name "${VIDEO_BASENAME}_results_*.json" -name "*_eval_results.json" -not -name "*_full_info.json" | sort -r | head -n 1)

    if [ -z "${VBENCH_RESULT_FILE}" ]; then
        echo -e "\n[Error] VBench result file not found!"
        # Depending on your preference, you can exit or continue
        # exit 1 
    else
        echo "[Success] VBench result file found: ${VBENCH_RESULT_FILE}"
    fi
else
    echo "[Warning] VBench script not found at ${VBENCH_EVAL_SCRIPT}. Skipping evaluation."
fi

# ============================== Step 4: Merge Results =================================
CURRENT_STEP=$( [ "$PATTERN" = "radial" ] && echo "4" || echo "3" )
echo -e "\n[Step ${CURRENT_STEP}/${TOTAL_STEPS}] Merging parameters, core metrics, and VBench results..."

FULL_RESULT_FILE="${EVAL_OUTPUT_DIR}/${VIDEO_BASENAME}_full_results_${TIMESTAMP}.json"
echo "Full results output path: ${FULL_RESULT_FILE}"

if [ -f "${MERGE_SCRIPT_PATH}" ] && [ -n "${VBENCH_RESULT_FILE}" ]; then
    python ${MERGE_SCRIPT_PATH} \
        --radial_params "${RADIAL_PARAMS_FILE}" \
        --vbench_results "${VBENCH_RESULT_FILE}" \
        --full_results "${FULL_RESULT_FILE}"

    if [ -f "${FULL_RESULT_FILE}" ]; then
        echo "[Success] Results merged successfully!"
    else
        echo "[Error] Result merging failed!"
    fi
else
    echo "[Warning] Merge script or VBench results missing. Skipping merge."
fi

# ============================== Pipeline End ==============================
echo -e "\n========================================"
echo "Pipeline Execution Completed! File Manifest:"
echo "1. Generated Video: ${VIDEO_OUTPUT_PATH}"
if [ "${PATTERN}" = "radial" ]; then
    echo "2. Mask File: ${MASK_FILE_PATH}"
    echo "3. Mask Visualization: ${TARGET_VIS_FILE}"
fi
echo "$( [ "$PATTERN" = "radial" ] && echo "4" || echo "2" ). Parameters & Core Metrics: ${RADIAL_PARAMS_FILE}"
echo "$( [ "$PATTERN" = "radial" ] && echo "5" || echo "3" ). VBench Scores: ${VBENCH_RESULT_FILE:-N/A}"
echo "$( [ "$PATTERN" = "radial" ] && echo "6" || echo "4" ). Merged Results: ${FULL_RESULT_FILE:-N/A}"
echo "========================================"