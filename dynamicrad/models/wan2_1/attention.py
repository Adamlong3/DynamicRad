from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    _get_qkv_projections,
    _get_added_kv_projections,
)
from einops import rearrange
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.distributed as dist

# Import our core DynamicRad sparse attention implementation
from ...dynamicrad.attention.attn_mask import RadialAttention

try:
    from xfuser.core.distributed import get_ulysses_parallel_world_size
    from xfuser.model_executor.layers.usp import (
        _ft_c_input_all_to_all,
        _ft_c_output_all_to_all,
    )
except ImportError:
    pass


class WanSparseAttnProcessor:
    """
    Custom Attention Processor for Wan2.1-14B to inject DynamicRad.
    """
    _attention_backend = None
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"
    use_sage_attention = False
    mask_threshold = 0.6
    long_factor = 1.0
    col_density_threshold = 1/3
    block_size = 128
    
    # DynamicRad specific parameters
    topk_ratio1 = 0.3
    topk_ratio2 = 0.6
    topk_seed = 42
    num_heads_to_fuse = 2
    vis_frame_pair_limit = 38
    topk_mode = "dynamic_ratio"
    near_frame_threshold = 0.0
    far_frame_threshold = 0.0

    def __init__(self, layer_idx: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0 or higher."
            )
        self.layer_idx = layer_idx
        self.use_sp = False

        if dist.is_initialized():
            try:
                if get_ulysses_parallel_world_size() > 1:
                    self.use_sp = True
            except Exception:
                self.use_sp = False

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        numerical_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if attn.cross_attention_dim_head is not None:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        else:
            batch_size = query.shape[0]

            if self.use_sp:
                query = rearrange(query, "b s h d -> b h s d").contiguous()
                key = rearrange(key, "b s h d -> b h s d").contiguous()
                value = rearrange(value, "b s h d -> b h s d").contiguous()

                query = _ft_c_input_all_to_all(query)
                key = _ft_c_input_all_to_all(key)
                value = _ft_c_input_all_to_all(value)

                query = rearrange(query, "b h s d -> b s h d").contiguous()
                key = rearrange(key, "b h s d -> b s h d").contiguous()
                value = rearrange(value, "b h s d -> b s h d").contiguous()

            # DynamicRad Execution
            if (
                numerical_timestep is not None
                and (numerical_timestep < self.dense_timestep or self.layer_idx < self.dense_block)
            ) or self.sparse_type == "dense":
                hidden_states = RadialAttention(
                    query=query, key=key, value=value,
                    mask_map=self.mask_map,
                    sparsity_type="dense",
                    block_size=self.block_size, decay_factor=self.decay_factor,
                    model_type="wan",
                    pre_defined_mask=None,
                    use_sage_attention=self.use_sage_attention,
                    mask_threshold=self.mask_threshold, long_factor=self.long_factor,
                    col_density_threshold=self.col_density_threshold,
                    topk_ratio1=self.topk_ratio1, topk_ratio2=self.topk_ratio2,
                    topk_seed=self.topk_seed, num_heads_to_fuse=self.num_heads_to_fuse,
                    vis_frame_pair_limit=self.vis_frame_pair_limit, topk_mode=self.topk_mode,
                    near_frame_threshold=self.near_frame_threshold, far_frame_threshold=self.far_frame_threshold
                )
            else:
                hidden_states = RadialAttention(
                    query=query, key=key, value=value,
                    mask_map=self.mask_map,
                    sparsity_type="radial",
                    block_size=self.block_size, decay_factor=self.decay_factor,
                    model_type="wan",
                    pre_defined_mask=None,
                    use_sage_attention=self.use_sage_attention,
                    mask_threshold=self.mask_threshold, long_factor=self.long_factor,
                    col_density_threshold=self.col_density_threshold,
                    topk_ratio1=self.topk_ratio1, topk_ratio2=self.topk_ratio2,
                    topk_seed=self.topk_seed, num_heads_to_fuse=self.num_heads_to_fuse,
                    vis_frame_pair_limit=self.vis_frame_pair_limit, topk_mode=self.topk_mode,
                    near_frame_threshold=self.near_frame_threshold, far_frame_threshold=self.far_frame_threshold
                )

            if self.use_sp:
                hidden_states = rearrange(hidden_states.contiguous(), "b s h d -> b h s d").contiguous()
                hidden_states = _ft_c_output_all_to_all(hidden_states)
                hidden_states = rearrange(hidden_states, "b h s d -> b s h d", b=batch_size).contiguous()

        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states