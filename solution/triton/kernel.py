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


@triton.jit
def _moe_gemm1_swiglu_kernel(
    hs_ptr: tl.pointer_type(tl.float8e4nv),
    hs_scale_ptr: tl.pointer_type(tl.float32),
    tok_idx_ptr: tl.pointer_type(tl.int32),
    Tk,
    w13_ptr: tl.pointer_type(tl.float8e4nv),
    s13_ptr: tl.pointer_type(tl.float32),
    c_ptr: tl.pointer_type(tl.float32),
    stride_hs_t,
    stride_hs_h,
    stride_hs_scale_hb,
    stride_hs_scale_t,
    stride_w13_o,
    stride_w13_h,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Tk
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    tok_idx = tl.load(tok_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    u1 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    u2 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    zero_a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float8e4nv)

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

        w1_ptrs = w13_ptr + offs_i[:, None] * stride_w13_o + offs_k[None, :] * stride_w13_h
        w2_ptrs = (
            w13_ptr
            + (I + offs_i)[:, None] * stride_w13_o
            + offs_k[None, :] * stride_w13_h
        )
        w1 = tl.load(w1_ptrs).to(tl.float32)
        w2 = tl.load(w2_ptrs).to(tl.float32)
        s1 = tl.load(s13_ptr + pid_i * stride_s13_o + kb * stride_s13_hb)
        s2 = tl.load(
            s13_ptr + (NUM_I_BLOCKS_C + pid_i) * stride_s13_o + kb * stride_s13_hb
        )

        u1 += tl.dot(a, tl.trans(w1 * s1))
        u2 += tl.dot(a, tl.trans(w2 * s2))

    silu_u2 = u2 / (1.0 + tl.exp(-u2))
    c = silu_u2 * u1
    c_ptrs = c_ptr + offs_m[:, None] * stride_c_t + offs_i[None, :] * stride_c_i
    tl.store(c_ptrs, c, mask=mask_m[:, None])


@triton.jit
def _moe_gemm2_accum_kernel(
    c_ptr: tl.pointer_type(tl.float32),
    tok_idx_ptr: tl.pointer_type(tl.int32),
    w_tok_ptr: tl.pointer_type(tl.float32),
    Tk,
    w2_ptr: tl.pointer_type(tl.float8e4nv),
    s2_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    stride_c_t,
    stride_c_i,
    stride_w2_h,
    stride_w2_i,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < Tk
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)

    tok_idx = tl.load(tok_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    w_tok = tl.load(w_tok_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ib in range(0, NUM_I_BLOCKS_C):
        offs_i = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        c_ptrs = c_ptr + offs_m[:, None] * stride_c_t + offs_i[None, :] * stride_c_i
        c = tl.load(c_ptrs, mask=mask_m[:, None], other=0.0)

        w2_ptrs = w2_ptr + offs_h[:, None] * stride_w2_h + offs_i[None, :] * stride_w2_i
        w = tl.load(w2_ptrs).to(tl.float32)
        scale = tl.load(s2_ptr + pid_h * stride_s2_hb + ib * stride_s2_ib)
        acc += tl.dot(c, tl.trans(w * scale))

    acc = acc * w_tok[:, None]
    out_ptrs = out_ptr + tok_idx[:, None] * stride_out_t + offs_h[None, :] * stride_out_h
    old = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0)
    tl.store(out_ptrs, old + acc, mask=mask_m[:, None])


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _as_python_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


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

    assert num_experts == NUM_EXPERTS_GLOBAL
    assert hidden_states.shape == (seq_len, HIDDEN_SIZE)
    assert hidden_states_scale.shape == (NUM_H_BLOCKS, seq_len)
    assert gemm1_weights.shape == (local_num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert gemm1_weights_scale.shape == (local_num_experts, 2 * NUM_I_BLOCKS, NUM_H_BLOCKS)
    assert gemm2_weights.shape == (local_num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    assert gemm2_weights_scale.shape == (local_num_experts, NUM_H_BLOCKS, NUM_I_BLOCKS)
    assert routing_bias is None or routing_bias.shape[-1] == NUM_EXPERTS_GLOBAL
    assert output.shape == (seq_len, HIDDEN_SIZE)

    local_expert_offset = _as_python_int(local_expert_offset)
    routed_scaling_factor = _as_python_float(routed_scaling_factor)

    routing_logits = routing_logits.to(torch.float32).contiguous()
    if routing_bias is None:
        routing_bias_f32 = torch.zeros(
            (NUM_EXPERTS_GLOBAL,), dtype=torch.float32, device=routing_logits.device
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
    pruned = scores_with_bias.masked_fill(expert_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    route_mask = torch.zeros_like(scores)
    route_mask.scatter_(1, topk_idx, 1.0)
    route_weights = scores * route_mask
    route_weights = route_weights / (route_weights.sum(dim=1, keepdim=True) + 1.0e-20)
    route_weights = route_weights * routed_scaling_factor

    out_accum = torch.zeros((seq_len, HIDDEN_SIZE), dtype=torch.float32, device=output.device)

    block_m = 32
    block_n = BLOCK_SIZE
    block_k = BLOCK_SIZE
    block_i = BLOCK_SIZE

    stride_hs_t = hidden_states.stride(0)
    stride_hs_h = hidden_states.stride(1)
    stride_hs_scale_hb = hidden_states_scale.stride(0)
    stride_hs_scale_t = hidden_states_scale.stride(1)
    stride_out_t = out_accum.stride(0)
    stride_out_h = out_accum.stride(1)

    for local_expert in range(local_num_experts):
        global_expert = local_expert_offset + local_expert
        if global_expert < 0 or global_expert >= NUM_EXPERTS_GLOBAL:
            continue

        selected = (topk_idx == global_expert).any(dim=1)
        if not torch.any(selected):
            continue

        tok_idx = torch.nonzero(selected, as_tuple=False).squeeze(1).to(torch.int32).contiguous()
        tk_local = int(tok_idx.numel())
        w_tok = (
            route_weights.index_select(0, tok_idx.to(torch.int64))[:, global_expert]
            .to(torch.float32)
            .contiguous()
        )
        c_buf = torch.empty(
            (tk_local, INTERMEDIATE_SIZE), dtype=torch.float32, device=output.device
        )

        w13_e = gemm1_weights[local_expert]
        s13_e = gemm1_weights_scale[local_expert]
        w2_e = gemm2_weights[local_expert]
        s2_e = gemm2_weights_scale[local_expert]

        grid_gemm1 = (triton.cdiv(tk_local, block_m), NUM_I_BLOCKS)
        _moe_gemm1_swiglu_kernel[grid_gemm1](
            hidden_states,
            hidden_states_scale,
            tok_idx,
            tk_local,
            w13_e,
            s13_e,
            c_buf,
            stride_hs_t,
            stride_hs_h,
            stride_hs_scale_hb,
            stride_hs_scale_t,
            w13_e.stride(0),
            w13_e.stride(1),
            s13_e.stride(0),
            s13_e.stride(1),
            c_buf.stride(0),
            c_buf.stride(1),
            INTERMEDIATE_SIZE,
            NUM_H_BLOCKS,
            NUM_I_BLOCKS,
            block_m,
            block_k,
            block_i,
            num_warps=8,
            num_stages=3,
        )

        grid_gemm2 = (triton.cdiv(tk_local, block_m), NUM_H_BLOCKS)
        _moe_gemm2_accum_kernel[grid_gemm2](
            c_buf,
            tok_idx,
            w_tok,
            tk_local,
            w2_e,
            s2_e,
            out_accum,
            c_buf.stride(0),
            c_buf.stride(1),
            w2_e.stride(0),
            w2_e.stride(1),
            s2_e.stride(0),
            s2_e.stride(1),
            stride_out_t,
            stride_out_h,
            NUM_I_BLOCKS,
            block_m,
            block_n,
            block_i,
            num_warps=8,
            num_stages=3,
        )

    output.copy_(out_accum.to(output.dtype))
