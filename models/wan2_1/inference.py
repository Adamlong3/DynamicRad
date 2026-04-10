import torch
import logging
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import AttentionModuleMixin
from .attention import WanSparseAttnProcessor
from ...dynamicrad.attention.attn_mask import MaskMap

logger = logging.getLogger(__name__)

def replace_wan_attention(
    pipe,
    height,
    width,
    num_frames,
    dense_layers=0,
    dense_timesteps=0,
    decay_factor=1.0,
    sparsity_type="radial",
    use_sage_attention=False,
    mask_threshold=0.6,
    long_factor=1.0,
    col_density_threshold=1/3,
    block_size=128,
    topk_ratio1=0.3,  
    topk_ratio2=0.6,  
    topk_seed=42,
    topk_mode="dynamic_ratio",
    near_frame_threshold=0.0,
    far_frame_threshold=0.0
):
    """
    Injects DynamicRad attention processor into the Wan2.1-14B diffusion pipeline.
    """
    # Calculate sequence dimensions
    num_frames = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_size = int(height // mod_value) * int(width // mod_value)
    
    # Configure WanSparseAttnProcessor
    AttnModule = WanSparseAttnProcessor
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.mask_map = MaskMap(video_token_num=frame_size * num_frames, num_frame=num_frames)
    AttnModule.decay_factor = decay_factor
    AttnModule.sparse_type = sparsity_type
    AttnModule.use_sage_attention = use_sage_attention
    AttnModule.mask_threshold = mask_threshold
    AttnModule.long_factor = long_factor
    AttnModule.col_density_threshold = col_density_threshold
    AttnModule.block_size = block_size  
    AttnModule.topk_ratio1 = topk_ratio1  
    AttnModule.topk_ratio2 = topk_ratio2  
    AttnModule.topk_seed = topk_seed
    AttnModule.topk_mode = topk_mode
    AttnModule.near_frame_threshold = near_frame_threshold
    AttnModule.far_frame_threshold = far_frame_threshold
    AttnModule.num_heads_to_fuse = 2
    AttnModule.vis_frame_pair_limit = 38

    logger.info(f"Injecting {sparsity_type} attention ({topk_mode} mode)")
    logger.info(f"Video tokens: {AttnModule.mask_map.video_token_num}, Frames: {num_frames}")

    # Inject processor into all transformer blocks
    replaced_count = 0
    total_blocks = len(pipe.transformer.blocks)
    for layer_idx, m in enumerate(pipe.transformer.blocks):
        m.attn1.set_processor(AttnModule(layer_idx))
        replaced_count += 1

    # Validate Injection
    validate_wan_attention_config(pipe)
    
    logger.info("MaskMap initialized. Sparse masks will be generated dynamically during the first inference step.")
    return None

def validate_wan_attention_config(pipe):
    """Verifies that all attention layers were successfully replaced."""
    replaced_layers = []
    for layer_idx, m in enumerate(pipe.transformer.blocks):
        if isinstance(m.attn1.processor, WanSparseAttnProcessor):
            replaced_layers.append(layer_idx)
    
    if len(replaced_layers) == len(pipe.transformer.blocks):
        logger.info(f"Successfully replaced all {len(replaced_layers)} attention layers.")
        return True
    else:
        missing_layers = [i for i in range(len(pipe.transformer.blocks)) if i not in replaced_layers]
        raise RuntimeError(f"Injection failed! The following layers were not replaced: {missing_layers}")