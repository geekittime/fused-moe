import torch
import triton
import triton.language as tl


NUM_EXPERTS_GLOBAL = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = 128
NUM_H_BLOCKS = HIDDEN_SIZE // BLOCK_SIZE
NUM_I_BLOCKS = INTERMEDIATE_SIZE // BLOCK_SIZE


# ═══════════════════════════════════════════════════════════════════════
# Fused GEMM1 + SwiGLU — all experts in ONE launch (no race on c_buf)
# ═══════════════════════════════════════════════════════════════════════
@triton.jit
def _fused_moe_gemm1_swiglu_kernel(
    hs_ptr: tl.pointer_type(tl.float8e4nv),
    hs_scale_ptr: tl.pointer_type(tl.float32),
    flat_tok_ptr: tl.pointer_type(tl.int32),
    block_offsets_ptr: tl.pointer_type(tl.int32),
    block_experts_ptr: tl.pointer_type(tl.int32),
    block_counts_ptr: tl.pointer_type(tl.int32),
    w13_ptr: tl.pointer_type(tl.float8e4nv),
    s13_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    stride_hs_t,
    stride_hs_h,
    stride_hs_scale_hb,
    stride_hs_scale_t,
    stride_w13_e,
    stride_w13_o,
    stride_w13_h,
    stride_s13_e,
    stride_s13_o,
    stride_s13_hb,
    stride_c_t,
    stride_c_i,
    I: tl.constexpr,
    NUM_H_BLOCKS_C: tl.constexpr,
    NUM_I_BLOCKS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    flat_off = tl.load(block_offsets_ptr + pid_m).to(tl.int64)
    expert  = tl.load(block_experts_ptr + pid_m).to(tl.int64)
    count   = tl.load(block_counts_ptr  + pid_m)

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < count
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    tok_idx = tl.load(flat_tok_ptr + flat_off + offs_m, mask=mask_m, other=0).to(tl.int32)

    u1 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    u2 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    zero_a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float8e4nv)

    w13_base = w13_ptr + expert * stride_w13_e
    s13_base = s13_ptr + expert * stride_s13_e

    for kb in range(0, NUM_H_BLOCKS_C):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = hs_ptr + tok_idx[:, None] * stride_hs_t + offs_k[None, :] * stride_hs_h
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=zero_a).to(tl.float32)
        a_scale = tl.load(
            hs_scale_ptr + kb * stride_hs_scale_hb + tok_idx * stride_hs_scale_t,
            mask=mask_m,
            other=0.0,
        )
        a = a * a_scale[:, None]

        w1_ptrs = w13_base + offs_i[:, None] * stride_w13_o + offs_k[None, :] * stride_w13_h
        w2_ptrs = (
            w13_base
            + (I + offs_i)[:, None] * stride_w13_o
            + offs_k[None, :] * stride_w13_h
        )
        w1 = tl.load(w1_ptrs).to(tl.float32)
        w2 = tl.load(w2_ptrs).to(tl.float32)
        s1 = tl.load(s13_base + pid_i * stride_s13_o + kb * stride_s13_hb)
        s2 = tl.load(
            s13_base + (NUM_I_BLOCKS_C + pid_i) * stride_s13_o + kb * stride_s13_hb
        )

        u1 += tl.dot(a, tl.trans(w1 * s1))
        u2 += tl.dot(a, tl.trans(w2 * s2))

    silu_u2 = u2 / (1.0 + tl.exp(-u2))
    c = silu_u2 * u1

    c_offs = flat_off + offs_m
    c_ptrs = c_ptr + c_offs[:, None] * stride_c_t + offs_i[None, :] * stride_c_i
    tl.store(c_ptrs, c, mask=mask_m[:, None])


# ═══════════════════════════════════════════════════════════════════════
# Fused GEMM2 + accumulate — all experts, atomic_add avoids race
# ═══════════════════════════════════════════════════════════════════════
@triton.jit
def _fused_moe_gemm2_accum_kernel(
    c_ptr: tl.pointer_type(tl.float32),
    flat_tok_ptr: tl.pointer_type(tl.int32),
    flat_w_ptr: tl.pointer_type(tl.float32),
    block_offsets_ptr: tl.pointer_type(tl.int32),
    block_experts_ptr: tl.pointer_type(tl.int32),
    block_counts_ptr: tl.pointer_type(tl.int32),
    w2_ptr: tl.pointer_type(tl.float8e4nv),
    s2_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    stride_c_t,
    stride_c_i,
    stride_w2_e,
    stride_w2_h,
    stride_w2_i,
    stride_s2_e,
    stride_s2_hb,
    stride_s2_ib,
    stride_out_t,
    stride_out_h,
    NUM_I_BLOCKS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    flat_off = tl.load(block_offsets_ptr + pid_m).to(tl.int64)
    expert  = tl.load(block_experts_ptr + pid_m).to(tl.int64)
    count   = tl.load(block_counts_ptr  + pid_m)

    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < count
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)

    tok_idx = tl.load(flat_tok_ptr + flat_off + offs_m, mask=mask_m, other=0).to(tl.int32)
    w_tok   = tl.load(flat_w_ptr   + flat_off + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w2_base = w2_ptr + expert * stride_w2_e
    s2_base = s2_ptr + expert * stride_s2_e

    c_offs = flat_off + offs_m
    for ib in range(0, NUM_I_BLOCKS_C):
        offs_i = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        c_ptrs = c_ptr + c_offs[:, None] * stride_c_t + offs_i[None, :] * stride_c_i
        c = tl.load(c_ptrs, mask=mask_m[:, None], other=0.0)

        w2_ptrs = w2_base + offs_h[:, None] * stride_w2_h + offs_i[None, :] * stride_w2_i
        w = tl.load(w2_ptrs).to(tl.float32)
        scale = tl.load(s2_base + pid_h * stride_s2_hb + ib * stride_s2_ib)
        acc += tl.dot(c, tl.trans(w * scale))

    acc = acc * w_tok[:, None]

    # atomic_add: multiple experts may write to same token — no race
    out_ptrs = out_ptr + tok_idx[:, None] * stride_out_t + offs_h[None, :] * stride_out_h
    tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None])


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)

def _as_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


# ═══════════════════════════════════════════════════════════════════════
# Optimised run():  2 total kernel launches instead of 64
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
):
    seq_len, num_experts = routing_logits.shape
    local_num_experts = gemm1_weights.shape[0]
    device = output.device

    local_expert_offset = _as_python_int(local_expert_offset)
    routed_scaling_factor = _as_python_float(routed_scaling_factor)

    # ── 1. Routing (identical to original) ───────────────────────────
    routing_logits = routing_logits.to(torch.float32).contiguous()
    if routing_bias is None:
        routing_bias_f32 = torch.zeros(
            (NUM_EXPERTS_GLOBAL,), dtype=torch.float32, device=device
        )
    else:
        routing_bias_f32 = routing_bias.to(torch.float32).contiguous().reshape(-1)
    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.to(torch.float32).contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.to(torch.float32).contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    gemm2_weights_scale = gemm2_weights_scale.to(torch.float32).contiguous()

    scores = torch.sigmoid(routing_logits)
    scores_with_bias = scores + routing_bias_f32

    group_size = NUM_EXPERTS_GLOBAL // N_GROUP
    grouped = scores_with_bias.view(seq_len, N_GROUP, group_size)
    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, top_group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    )

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, top_group_idx, 1.0)
    expert_mask = (
        group_mask.unsqueeze(2)
        .expand(seq_len, N_GROUP, group_size)
        .reshape(seq_len, NUM_EXPERTS_GLOBAL)
    )
    pruned = scores_with_bias.masked_fill(
        expert_mask == 0, torch.finfo(torch.float32).min
    )
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    route_mask = torch.zeros_like(scores)
    route_mask.scatter_(1, topk_idx, 1.0)
    route_weights = scores * route_mask
    route_weights = route_weights / (
        route_weights.sum(dim=1, keepdim=True) + 1.0e-20
    )
    route_weights = route_weights * routed_scaling_factor

    # ── 2. Batch token→expert assignment ─────────────────────────────
    local_end = local_expert_offset + local_num_experts
    local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_end)

    tok_ids_flat, slot_ids_flat = torch.where(local_mask)
    total_assignments = tok_ids_flat.numel()

    out_accum = torch.zeros(
        (seq_len, HIDDEN_SIZE), dtype=torch.float32, device=device
    )

    if total_assignments == 0:
        output.copy_(out_accum.to(output.dtype))
        return

    global_eids = topk_idx[tok_ids_flat, slot_ids_flat]
    local_eids = (global_eids - local_expert_offset).to(torch.int64)

    # Sort by local expert so tokens per expert are contiguous
    sort_idx = torch.argsort(local_eids, stable=True)
    sorted_tok = tok_ids_flat[sort_idx].to(torch.int32).contiguous()
    sorted_local = local_eids[sort_idx]
    sorted_global = global_eids[sort_idx]

    sorted_weights = (
        route_weights[tok_ids_flat[sort_idx].to(torch.int64), sorted_global]
        .to(torch.float32)
        .contiguous()
    )

    # Expert boundaries
    expert_counts = torch.bincount(sorted_local, minlength=local_num_experts)
    expert_starts = torch.zeros(
        local_num_experts + 1, dtype=torch.int64, device=device
    )
    torch.cumsum(expert_counts, dim=0, out=expert_starts[1:])

    # ── 3. Build block metadata on CPU ───────────────────────────────
    ec_cpu = expert_counts.cpu()
    es_cpu = expert_starts.cpu()

    BLOCK_M = 64

    block_off_list = []
    block_exp_list = []
    block_cnt_list = []
    for e in range(local_num_experts):
        cnt = int(ec_cpu[e].item())
        if cnt == 0:
            continue
        start = int(es_cpu[e].item())
        for b in range(0, cnt, BLOCK_M):
            block_off_list.append(start + b)
            block_exp_list.append(e)
            block_cnt_list.append(min(BLOCK_M, cnt - b))

    n_blocks = len(block_off_list)
    if n_blocks == 0:
        output.copy_(out_accum.to(output.dtype))
        return

    block_offsets = torch.tensor(block_off_list, dtype=torch.int32, device=device)
    block_experts = torch.tensor(block_exp_list, dtype=torch.int32, device=device)
    block_counts  = torch.tensor(block_cnt_list, dtype=torch.int32, device=device)

    # ── 4. Allocate intermediate buffer ONCE ─────────────────────────
    c_buf = torch.empty(
        (total_assignments, INTERMEDIATE_SIZE), dtype=torch.float32, device=device
    )

    BLOCK_K = BLOCK_SIZE
    BLOCK_I = BLOCK_SIZE
    BLOCK_N = BLOCK_SIZE

    # ── 5. Fused GEMM1 + SwiGLU — ONE kernel launch ─────────────────
    grid_gemm1 = (n_blocks, NUM_I_BLOCKS)
    _fused_moe_gemm1_swiglu_kernel[grid_gemm1](
        hidden_states,
        hidden_states_scale,
        sorted_tok,
        block_offsets,
        block_experts,
        block_counts,
        gemm1_weights,
        gemm1_weights_scale,
        c_buf,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states_scale.stride(0),
        hidden_states_scale.stride(1),
        gemm1_weights.stride(0),
        gemm1_weights.stride(1),
        gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0),
        gemm1_weights_scale.stride(1),
        gemm1_weights_scale.stride(2),
        c_buf.stride(0),
        c_buf.stride(1),
        INTERMEDIATE_SIZE,
        NUM_H_BLOCKS,
        NUM_I_BLOCKS,
        BLOCK_M,
        BLOCK_K,
        BLOCK_I,
        num_warps=8,
        num_stages=3,
    )

    # ── 6. Fused GEMM2 + atomic accumulate — ONE kernel launch ───────
    grid_gemm2 = (n_blocks, NUM_H_BLOCKS)
    _fused_moe_gemm2_accum_kernel[grid_gemm2](
        c_buf,
        sorted_tok,
        sorted_weights,
        block_offsets,
        block_experts,
        block_counts,
        gemm2_weights,
        gemm2_weights_scale,
        out_accum,
        c_buf.stride(0),
        c_buf.stride(1),
        gemm2_weights.stride(0),
        gemm2_weights.stride(1),
        gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0),
        gemm2_weights_scale.stride(1),
        gemm2_weights_scale.stride(2),
        out_accum.stride(0),
        out_accum.stride(1),
        NUM_I_BLOCKS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_I,
        num_warps=8,
        num_stages=3,
    )

    output.copy_(out_accum.to(output.dtype))