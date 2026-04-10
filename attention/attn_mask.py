import torch
import flashinfer
import os
from einops import rearrange, repeat
from sageattention import sageattn

try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
except ImportError:
    print("Warning: Using sparse_sageattn as block_sparse_sage2_attn_cuda")
    from sparse_sageattn import sparse_sageattn as block_sparse_sage2_attn_cuda

# Default visualization directory
VIS_SAVE_DIR = "./visualizations"
os.makedirs(VIS_SAVE_DIR, exist_ok=True)


# ----------------- Utility Functions ----------------- #

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


def sparge_mask_convert(mask: torch.Tensor, block_size: int = 128, arch="sm") -> torch.Tensor:
    assert block_size in [128, 64, 32], "DynamicRad only supports block size of 32/64/128"
    assert mask.shape[0] == mask.shape[1], "Input mask must be square."

    if block_size == 128:
        if arch == "sm90":
            new_mask = torch.repeat_interleave(mask, 2, dim=0)
        else:
            new_mask = torch.repeat_interleave(mask, 2, dim=1)
    elif block_size == 64:
        num_row, num_col = mask.shape
        if arch == "sm90":
            reshaped_mask = mask.view(num_row, num_col // 2, 2)
            new_mask = torch.max(reshaped_mask, dim=2).values
        else:
            reshaped_mask = mask.view(num_row // 2, 2, num_col)
            new_mask = torch.max(reshaped_mask, dim=1).values
    elif block_size == 32:
        num_row, num_col = mask.shape
        if arch == "sm90":
            reshaped_mask = mask.view(num_row, num_col // 4, 4)
            new_mask = torch.max(reshaped_mask, dim=2).values
        else:
            reshaped_mask = mask.view(num_row // 4, 4, num_col)
            new_mask = torch.max(reshaped_mask, dim=1).values
    return new_mask


def get_indptr_from_mask(mask, query):
    indptr = torch.zeros(mask.shape[0] + 1, device=query.device, dtype=torch.int32)
    indptr[0] = 0
    row_counts = mask.sum(dim=1).flatten()
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return indptr


def get_indices_from_mask(mask, query):
    nonzero_indices = torch.nonzero(mask)
    indices = nonzero_indices[:, 1].to(dtype=torch.int32, device=query.device)
    return indices


def shrinkMaskStrict(mask, block_size=128, mask_threshold=0.6, col_density_threshold=1/3):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[:block_num * block_size, :block_num * block_size]
    mask = mask.view(block_num, block_size, block_num, block_size).permute(0, 2, 1, 3)
    col_densities = mask.sum(dim=2) / block_size
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > col_density_threshold
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high_density_cols > mask_threshold
    return block_mask


def pad_qkv(input_tensor, block_size=128):
    seqlen, num_heads, hidden_dim = input_tensor.shape
    padding_length = (block_size - (seqlen % block_size)) % block_size
    padded_tensor = torch.zeros(
        (seqlen + padding_length, num_heads, hidden_dim),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    padded_tensor[:seqlen, :, :] = input_tensor
    return padded_tensor


def get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query,
                            long_factor=1.0, block_size=128):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = block_size
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * long_factor
    
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame),
                          device=query.device, dtype=torch.bool)
    
    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    
    if modular == 0:
        return torch.ones((token_per_frame, token_per_frame),
                          device=query.device, dtype=torch.bool)
    else:
        return torch.zeros((token_per_frame, token_per_frame),
                           device=query.device, dtype=torch.bool)


def compute_window_attn_scores_fp16(
    query, key, token_per_frame, i, j, valid_pairs, num_heads_to_fuse=2
):
    """
    Computes proxy attention scores using a subset of heads to minimize overhead.
    """
    batch_size, _, num_head_total, hidden_dim = query.shape
    assert batch_size == 1, "Current implementation assumes batch_size=1"
    
    num_heads_to_fuse = min(num_heads_to_fuse, num_head_total)
    scale = torch.sqrt(torch.tensor(hidden_dim, device=query.device, dtype=torch.float16))

    num_frame = query.shape[1] // token_per_frame
    i = min(i, num_frame - 1)
    j = min(j, num_frame - 1)

    frame_i_start = i * token_per_frame
    frame_i_end = (i + 1) * token_per_frame
    frame_j_start = j * token_per_frame
    frame_j_end = (j + 1) * token_per_frame

    query_i = query[0, frame_i_start:frame_i_end, :num_heads_to_fuse, :].half()
    key_j = key[0, frame_j_start:frame_j_end, :num_heads_to_fuse, :].half()

    ti_global = valid_pairs[:, 0]
    tj_global = valid_pairs[:, 1]

    video_total_tokens = query.shape[1]
    ti_global = torch.clamp(ti_global, 0, video_total_tokens - 1)
    tj_global = torch.clamp(tj_global, 0, video_total_tokens - 1)

    ti_local = ti_global - frame_i_start
    tj_local = tj_global - frame_j_start
    ti_local = torch.clamp(ti_local, 0, token_per_frame - 1)
    tj_local = torch.clamp(tj_local, 0, token_per_frame - 1)

    head_scores_list = []
    for head_idx in range(num_heads_to_fuse):
        query_vecs = query_i[ti_local, head_idx, :]
        key_vecs = key_j[tj_local, head_idx, :]
        head_score = torch.sum(query_vecs * key_vecs, dim=-1) / scale
        head_scores_list.append(head_score)
        
    attn_scores = torch.stack(head_scores_list, dim=0).mean(dim=0).float()
    return attn_scores


def get_window_width(
    i, j, token_per_frame, sparse_type, num_frame, query_device,
    decay_factor=1, block_size=128, model_type=None,
    topk_mode="static_ratio",
    topk_ratio1=0.6, topk_ratio2=0.4,
    near_frame_threshold=0.0, far_frame_threshold=0.0,
):
    assert sparse_type in ["radial"], "Only support radial sparse type"
    dist = abs(i - j)
    threshold = block_size
    window_width = 0

    if model_type == "wan":
        if dist <= 1:
            window_width = token_per_frame
            current_param = 1.0 if topk_mode in ["static_ratio", "dynamic_ratio"] else -1e6
        else:
            group = dist.bit_length()
            decay_length = (2 ** token_per_frame.bit_length() / 2 ** group) * decay_factor
            if decay_length >= threshold:
                window_width = decay_length
                current_param = topk_ratio1 if topk_mode in ["static_ratio", "dynamic_ratio"] else near_frame_threshold
            else:
                window_width = threshold
                current_param = topk_ratio2 if topk_mode in ["static_ratio", "dynamic_ratio"] else far_frame_threshold

    elif model_type == "hunyuan":
        if dist <= 1:
            window_width = token_per_frame
            current_param = 1.0 if topk_mode in ["static_ratio", "dynamic_ratio"] else -1e6
        else:
            group = dist.bit_length()
            decay_length = (2 ** token_per_frame.bit_length() / 2 ** group) * decay_factor
            if decay_length >= threshold:
                window_width = decay_length
                current_param = topk_ratio1 if topk_mode in ["static_ratio", "dynamic_ratio"] else near_frame_threshold
            else:
                window_width = threshold
                current_param = topk_ratio2 if topk_mode in ["static_ratio", "dynamic_ratio"] else far_frame_threshold
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    local_i = torch.arange(0, token_per_frame, device=query_device)
    local_j = torch.arange(0, token_per_frame, device=query_device)
    local_ti_grid, local_tj_grid = torch.meshgrid(local_i, local_j, indexing="ij")

    valid_local_mask = torch.abs(local_ti_grid - local_tj_grid) <= window_width

    global_ti = i * token_per_frame + local_ti_grid
    global_tj = j * token_per_frame + local_tj_grid

    all_global_pairs = torch.stack([global_ti.flatten(), global_tj.flatten()], dim=1)
    valid_pairs = all_global_pairs[valid_local_mask.flatten()]
    total_valid = valid_pairs.shape[0]

    dummy_mask = torch.zeros(total_valid, dtype=torch.bool, device=query_device)
    return window_width, dummy_mask, valid_pairs, current_param


def gen_log_mask_shrinked(
    query, key,
    s, video_token_num, num_frame, block_size=128,
    sparse_type="radial", decay_factor=0.5, model_type=None,
    mask_threshold=0.6, long_factor=1.0, col_density_threshold=1/3,
    topk_mode="static_ratio", 
    topk_ratio1=0.3, topk_ratio2=0.6, topk_seed=0, 
    near_frame_threshold=0.0, far_frame_threshold=0.0, 
    num_heads_to_fuse=2
):
    """
    Core function for generating block-sparse masks based on DynamicRad modes.
    Supports 'static_ratio' and 'dynamic_threshold' strategies.
    """
    assert topk_mode in ["static_ratio", "dynamic_ratio", "dynamic_threshold"]
    batch_size = query.shape[0]
    assert batch_size == 1, "Current implementation assumes batch_size=1"

    final_log_mask = torch.zeros((s // block_size, s // block_size),
                                 device=query.device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_text_border = video_token_num // block_size
    query_device = query.device

    col_indices = torch.arange(0, token_per_frame, device=query_device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=query_device).view(-1, 1)

    # Dense connection for text regions
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True

    for i in range(num_frame):
        for j in range(num_frame):
            
            # -----------------------------------------------------------
            # Strictly maintain dense self-attention (diagonal)
            # Crucial to prevent spatial artifacts and video jittering
            # -----------------------------------------------------------
            if i == j:
                b_r_s = (i * token_per_frame) // block_size
                b_c_s = (j * token_per_frame) // block_size
                
                b_r_e = ( (i + 1) * token_per_frame + block_size - 1 ) // block_size
                b_c_e = ( (j + 1) * token_per_frame + block_size - 1 ) // block_size
                
                b_r_e = min(b_r_e, final_log_mask.shape[0])
                b_c_e = min(b_c_e, final_log_mask.shape[1])
                
                final_log_mask[b_r_s:b_r_e, b_c_s:b_c_e] = True
                continue  

            local_mask = torch.zeros((token_per_frame, token_per_frame),
                                     device=query_device, dtype=torch.bool)

            # WAN Specific: All frames attend to the 0-th frame
            if j == 0 and model_type == "wan":
                local_mask = torch.ones((token_per_frame, token_per_frame),
                                        device=query_device, dtype=torch.bool)
            else:
                window_width, _, valid_pairs_window, current_param = get_window_width(
                    i=i, j=j,
                    token_per_frame=token_per_frame,
                    sparse_type=sparse_type,
                    num_frame=num_frame,
                    query_device=query_device,
                    decay_factor=decay_factor,
                    block_size=block_size,
                    model_type=model_type,
                    topk_mode=topk_mode,
                    topk_ratio1=topk_ratio1,
                    topk_ratio2=topk_ratio2,
                    near_frame_threshold=near_frame_threshold,
                    far_frame_threshold=far_frame_threshold,
                )

                split_mask = get_diagonal_split_mask(
                    i, j, token_per_frame, sparse_type, query,
                    long_factor=long_factor, block_size=block_size
                )

                local_ti = valid_pairs_window[:, 0] % token_per_frame
                local_tj = valid_pairs_window[:, 1] % token_per_frame
                local_ti = torch.clamp(local_ti, 0, token_per_frame - 1)
                local_tj = torch.clamp(local_tj, 0, token_per_frame - 1)
                
                split_valid_mask = split_mask[local_ti, local_tj]
                valid_pairs = valid_pairs_window[split_valid_mask]
                total_valid = valid_pairs.shape[0]

                window_mask = torch.abs(col_indices - row_indices) <= window_width
                local_mask = torch.logical_and(window_mask, split_mask)

                # Core Mode Selection Logic
                if total_valid > 0:
                    if topk_mode == "static_ratio":
                        gen = torch.Generator(device=query_device)
                        gen.manual_seed(topk_seed + i * num_frame + j)
                        k = max(int(total_valid * current_param), block_size)
                        k = min(k, total_valid)
                        random_indices = torch.randperm(total_valid, generator=gen, device=query_device)
                        topk_positions = random_indices[:k]
                        reserved_pairs = valid_pairs[topk_positions]

                        local_ti_reserved = reserved_pairs[:, 0] % token_per_frame
                        local_tj_reserved = reserved_pairs[:, 1] % token_per_frame

                        token_pair_mask = torch.zeros_like(local_mask, dtype=torch.bool)
                        token_pair_mask[local_ti_reserved, local_tj_reserved] = True
                        local_mask = torch.logical_and(local_mask, token_pair_mask)

                    elif topk_mode == "dynamic_ratio":
                        attn_scores = compute_window_attn_scores_fp16(
                            query, key, token_per_frame, i, j, valid_pairs, num_heads_to_fuse
                        )

                        k = max(int(total_valid * current_param), block_size)
                        k = min(k, total_valid)
                        topk_indices = torch.argsort(attn_scores, descending=True)[:k]
                        reserved_pairs = valid_pairs[topk_indices]

                        local_ti_reserved = reserved_pairs[:, 0] % token_per_frame
                        local_tj_reserved = reserved_pairs[:, 1] % token_per_frame

                        token_pair_mask = torch.zeros_like(local_mask, dtype=torch.bool)
                        token_pair_mask[local_ti_reserved, local_tj_reserved] = True
                        local_mask = torch.logical_and(local_mask, token_pair_mask)

                    else:  # dynamic_threshold
                        attn_scores = compute_window_attn_scores_fp16(
                            query, key, token_per_frame, i, j, valid_pairs, num_heads_to_fuse
                        )

                        current_threshold = current_param
                        score_mask = attn_scores >= current_threshold
                        num_kept = score_mask.sum().item()

                        # Fallback mechanism if threshold is too aggressive
                        if num_kept == 0:
                            top_idx = torch.argmax(attn_scores)
                            score_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
                            score_mask[top_idx] = True

                        reserved_pairs = valid_pairs[score_mask]
                        local_ti_reserved = reserved_pairs[:, 0] % token_per_frame
                        local_tj_reserved = reserved_pairs[:, 1] % token_per_frame

                        token_pair_mask = torch.zeros_like(local_mask, dtype=torch.bool)
                        token_pair_mask[local_ti_reserved, local_tj_reserved] = True
                        local_mask = torch.logical_and(local_mask, token_pair_mask)

            # Token mask to Block mask aggregation
            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            
            padded_local_mask = torch.zeros((all_length_row, all_length_col),
                                            device=query_device, dtype=torch.bool)
            padded_local_mask[
                remainder_row:remainder_row + token_per_frame,
                remainder_col:remainder_col + token_per_frame
            ] = local_mask

            block_mask = shrinkMaskStrict(
                padded_local_mask,
                block_size=block_size,
                mask_threshold=mask_threshold,
                col_density_threshold=col_density_threshold
            )

            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]

            final_log_mask[
                block_row_start:block_row_end,
                block_col_start:block_col_end
            ] = torch.logical_or(
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end],
                block_mask
            )

    # Note: Torch.save functionality can be disabled in production to save I/O overhead.
    # def fmt(x): return f"{x:.3g}"
    # filename = f"bs{block_size}_df{fmt(decay_factor)}_lf{fmt(long_factor)}_mt{fmt(mask_threshold)}_{topk_mode}_mask.pt"
    # save_path = os.path.join(VIS_SAVE_DIR, filename)
    # torch.save(final_log_mask.cpu(), save_path)

    return final_log_mask


# ----------------- MaskMap Caching ----------------- #

class MaskMap:
    def __init__(self, video_token_num=25440, num_frame=16):
        self.video_token_num = video_token_num
        self.num_frame = num_frame
        self._log_mask = None 

    def queryLogMask(self, query, key, **kwargs):
        if self._log_mask is None:
            seq_len = query.shape[1]
            self._log_mask = gen_log_mask_shrinked(
                query=query, key=key,
                s=seq_len, video_token_num=self.video_token_num, num_frame=self.num_frame,
                **kwargs
            )
        return self._log_mask


# ----------------- Attention Backends ----------------- #

def RadialAttention(
    query, key, value, mask_map=None, sparsity_type="radial",
    block_size=128, decay_factor=1, model_type=None, pre_defined_mask=None,
    use_sage_attention=False, mask_threshold=0.6, long_factor=1.0,
    col_density_threshold=1/3, topk_mode="static_ratio",                 
    topk_ratio1=0.3, topk_ratio2=0.6, topk_seed=0,
    near_frame_threshold=0.0, far_frame_threshold=0.0,
    num_heads_to_fuse=2
):
    batch_size, seq_len, num_head, hidden_dim = query.shape
    assert batch_size == 1, "Current implementation assumes batch_size=1"

    if sparsity_type == "dense":
        video_mask = torch.ones(
            (mask_map.video_token_num // block_size, mask_map.video_token_num // block_size),
            device=query.device, dtype=torch.bool
        )
    else:
        video_mask = mask_map.queryLogMask(
            query=query, key=key,
            sparse_type=sparsity_type, block_size=block_size,
            decay_factor=decay_factor, model_type=model_type,
            mask_threshold=mask_threshold, long_factor=long_factor,
            col_density_threshold=col_density_threshold,
            topk_mode=topk_mode,
            topk_ratio1=topk_ratio1, topk_ratio2=topk_ratio2, topk_seed=topk_seed,
            near_frame_threshold=near_frame_threshold, far_frame_threshold=far_frame_threshold,
            num_heads_to_fuse=num_heads_to_fuse
        ) if mask_map else None

    backend = "sparse_sageattn" if use_sage_attention else "flashinfer"

    if backend == "flashinfer":
        video_mask = video_mask[:mask_map.video_token_num // block_size,
                                :mask_map.video_token_num // block_size]
        workspace_buffer = torch.empty(128 * 1024 * 1024,
                                       device=query.device, dtype=torch.uint8)
        bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(
            workspace_buffer,
            backend="fa2",
        )

        indptr = get_indptr_from_mask(video_mask, query)
        indices = get_indices_from_mask(video_mask, query)

        bsr_wrapper.plan(
            indptr=indptr, indices=indices,
            M=mask_map.video_token_num, N=mask_map.video_token_num,
            R=block_size, C=block_size,
            num_qo_heads=num_head, num_kv_heads=num_head,
            head_dim=hidden_dim,
            q_data_type=query.dtype, kv_data_type=key.dtype, o_data_type=query.dtype,
        )

        return FlashInferBackend(query, key, value, mask_map, pre_defined_mask, bsr_wrapper, block_size=block_size)
    else:
        return SpargeSageAttnBackend(query, key, value, mask_map, video_mask, pre_defined_mask, block_size=block_size)


def SpargeSageAttnBackend(query, key, value, mask_map=None, video_mask=None,
                          pre_defined_mask=None, block_size=128):
    batch_size = query.shape[0]

    if video_mask.all():
        kv_border = pre_defined_mask[0].sum() if pre_defined_mask is not None else key.shape[1]
        output_video = sageattn(
            query[:, :mask_map.video_token_num, :, :],
            key[:, :kv_border, :, :],
            value[:, :kv_border, :, :],
            tensor_layout="NHD",
        )

        if pre_defined_mask is not None:
            q_flashinfer = rearrange(query[:, mask_map.video_token_num:, :, :], "b s h d -> (b s) h d")
            k_flashinfer = rearrange(key[:, :pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
            v_flashinfer = rearrange(value[:, :pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
            output_text = flashinfer.single_prefill_with_kv_cache(
                q=q_flashinfer, k=k_flashinfer, v=v_flashinfer,
                causal=False, return_lse=False,
            )
            output_text = rearrange(output_text, "(b s) h d -> b s (h d)", b=batch_size)
            output_video_flat = output_video.flatten(2, 3)
            return torch.cat([output_video_flat, output_text], dim=1)
        else:
            return output_video.flatten(2, 3)

    arch = get_cuda_arch_versions()[query.device.index]
    converted_mask = repeat(
        sparge_mask_convert(mask=video_mask, block_size=block_size, arch=arch),
        "s t -> b h s t", b=batch_size, h=query.shape[2]
    )
    converted_mask = converted_mask.to(torch.int8)

    query_bhsd = rearrange(query, "b s h d -> b h s d")
    key_bhsd = rearrange(key, "b s h d -> b h s d")
    value_bhsd = rearrange(value, "b s h d -> b h s d")

    if pre_defined_mask is None:
        output = block_sparse_sage2_attn_cuda(
            query_bhsd, key_bhsd, value_bhsd,
            mask_id=converted_mask[:, :, :, :key_bhsd.shape[2] // block_size].contiguous(),
            tensor_layout="HND",
        )
        return rearrange(output, "b h s d -> b s (h d)")

    kv_border = (pre_defined_mask[0].sum() + 63) // 64
    converted_mask[:, :, :, kv_border:] = False
    output_video = block_sparse_sage2_attn_cuda(
        query_bhsd[:, :, :mask_map.video_token_num, :],
        key_bhsd, value_bhsd,
        mask_id=converted_mask[:, :, :mask_map.video_token_num // block_size, :].contiguous(),
        tensor_layout="HND",
    )
    output_video = rearrange(output_video, "b h s d -> b s (h d)")

    q_flashinfer = rearrange(query[:, mask_map.video_token_num:, :, :], "b s h d -> (b s) h d")
    k_flashinfer = rearrange(key[:, :pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
    v_flashinfer = rearrange(value[:, :pre_defined_mask[0].sum(), :, :], "b s h d -> (b s) h d")
    output_text = flashinfer.single_prefill_with_kv_cache(
        q=q_flashinfer, k=k_flashinfer, v=v_flashinfer,
        causal=False, return_lse=False,
    )
    output_text = rearrange(output_text, "(b s) h d -> b s (h d)", b=batch_size)

    return torch.cat([output_video, output_text], dim=1)


def FlashInferBackend(query, key, value, mask_map=None,
                      pre_defined_mask=None, bsr_wrapper=None, block_size=128):
    batch_size = query.shape[0]
    query_shd = rearrange(query, "b s h d -> (b s) h d")
    key_shd = rearrange(key, "b s h d -> (b s) h d")
    value_shd = rearrange(value, "b s h d -> (b s) h d")

    if pre_defined_mask is not None:
        video_video_o, video_video_o_lse = bsr_wrapper.run(
            query_shd[:mask_map.video_token_num, :, :],
            key_shd[:mask_map.video_token_num, :, :],
            value_shd[:mask_map.video_token_num, :, :],
            return_lse=True
        )
        video_text_o, video_text_o_lse = flashinfer.single_prefill_with_kv_cache(
            q=query_shd[:mask_map.video_token_num, :, :],
            k=key_shd[mask_map.video_token_num:, :, :],
            v=value_shd[mask_map.video_token_num:, :, :],
            causal=False, return_lse=True,
            custom_mask=pre_defined_mask[:mask_map.video_token_num, mask_map.video_token_num:]
        )

        o_video, _ = flashinfer.merge_state(
            v_a=video_video_o, s_a=video_video_o_lse,
            v_b=video_text_o, s_b=video_text_o_lse
        )

        o_text = flashinfer.single_prefill_with_kv_cache(
            q=query_shd[mask_map.video_token_num:, :, :],
            k=key_shd, v=value_shd,
            causal=False, return_lse=False,
            custom_mask=pre_defined_mask[mask_map.video_token_num:, :]
        )

        output = torch.cat([o_video, o_text], dim=0)
        return rearrange(output, "(b s) h d -> b s (h d)", b=batch_size)
    else:
        o = bsr_wrapper.run(
            query_shd[:mask_map.video_token_num, :, :],
            key_shd[:mask_map.video_token_num, :, :],
            value_shd[:mask_map.video_token_num, :, :]
        )
        return rearrange(o, "(b s) h d -> b s (h d)", b=batch_size)


if __name__ == "__main__":
    # Simple sanity check for mask generation
    query = torch.randn(1, 38 * 640, 4, 64).cuda()
    key = torch.randn(1, 38 * 640, 4, 64).cuda()
    video_token_num = 38 * 640
    num_frame = 38
    block_size = 32
    padded_video_token_num = (video_token_num + block_size - 1) // block_size * block_size

    print(f"Testing mask generation...")
    temporal_mask = gen_log_mask_shrinked(
        query=query, key=key,
        s=padded_video_token_num, video_token_num=video_token_num, num_frame=num_frame,
        sparse_type="radial", decay_factor=0.7, model_type="hunyuan",
        mask_threshold=0.6, long_factor=1.0, col_density_threshold=0.5,
        block_size=block_size,
        topk_mode="dynamic_threshold",
        topk_ratio1=0.6, topk_ratio2=0.4, topk_seed=42,
        near_frame_threshold=0.0, far_frame_threshold=0.0,
        num_heads_to_fuse=2
    )

    print(f"Mask shape: {temporal_mask.shape}, Sparsity: {1 - temporal_mask.float().mean().item():.4f}")
    print("Test passed successfully.")