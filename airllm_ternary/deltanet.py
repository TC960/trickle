"""Gated DeltaNet: a dependency-free reimplementation of Qwen3.5/3.6-MoE's
linear-attention layer, for this project's own streaming engine.

Why this exists instead of just using `transformers`' own class directly:
Qwen/Qwen3.5-35B-A3B (config: `Qwen3_5MoeForConditionalGeneration`) alternates
3 `linear_attention` (Gated DeltaNet) layers with 1 `full_attention` layer,
repeating for all 40 layers (`config.text_config.layer_types`). transformers
ships a native implementation
(`transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet`)
but it prefers optimized kernels from the `causal_conv1d` and `fla` packages
when installed, falling back to a pure-PyTorch path
(`torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule` /
`causal_conv1d_fn` / `causal_conv1d_update`) when they are not. Those optional
packages ship prebuilt CUDA extensions with no ARM/Jetson wheels, so relying on
them would silently make this engine unable to run on the target device the
moment someone `pip install`s them on a dev box. This module vendors exactly
that pure-PyTorch fallback path -- ported line-for-line from
`transformers==5.16.1`'s installed fallback functions -- as a standalone,
always-available implementation this engine owns outright.

Numerics ported here (see the module docstrings below for line-level parity
notes against the reference):
  - `causal_conv1d_fn` / `causal_conv1d_update` -- the short causal depthwise
    conv over the QKV(+gate) projection, prefill and single-step decode forms.
  - `recurrent_gated_delta_rule` -- the decode-time per-token recurrence. State
    shape `[batch, num_v_heads, k_head_dim, v_head_dim]`, does NOT grow with
    sequence length -- structurally a KV-cache entry, computed in float32
    regardless of the surrounding model's dtype (matches `mamba_ssm_dtype:
    float32` in the real Qwen3.5-35B-A3B config.json).
  - `chunk_gated_delta_rule` -- the parallel prefill form (chunked UT-transform
    + sequential chunk scan). Ported in full; NOT stubbed out.
  - `RMSNormGated`, `GatedDeltaNet` -- the surrounding module: input
    projections, causal conv, gated RMSNorm, output projection.

`GatedDeltaNet.forward` intentionally keeps the exact same call contract as
the reference (`hidden_states`, `cache_params`, `attention_mask`, and the
`cache_params.has_previous_state / .layers[i].conv_states / .recurrent_states
/ .update_conv_state / .update_recurrent_state` protocol). That protocol is
implemented generically by `transformers.cache_utils.Cache` +
`LinearAttentionLayer`, which already ships in this project's `transformers`
dependency and already handles both linear-attention and full-attention state
side by side in one cache object. We deliberately do NOT reinvent a state-cache
class here: reusing the real one means a `GatedDeltaNet` instance is a
drop-in replacement for the reference module and needs no new plumbing in
`loader.py` -- see the module docstring in `model.py` for why.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal activation registry. `hidden_act` is "silu" for every published
# Qwen3.5-MoE config; the others are here only so a differently-configured
# checkpoint doesn't hard-fail.
_ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
}


def _act(name: str):
    try:
        return _ACT2FN[name]
    except KeyError:
        raise ValueError(f"unsupported activation {name!r}; add it to _ACT2FN in deltanet.py")


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """Zero out padding-token positions before the causal conv sees them.

    Ported verbatim from `modeling_qwen3_5_moe.apply_mask_to_padding_states`.
    See https://github.com/state-spaces/mamba/issues/66 for why this matters:
    a short causal conv leaks left-context across a padding boundary that
    ordinary attention masking would otherwise hide.
    """
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Single-token decode step of the causal depthwise conv.

    `conv_state` is updated in place (rolling window of the last
    `kernel_size` raw inputs) and the function returns just the one new
    output position. Ported verbatim from
    `modeling_qwen3_5_moe.causal_conv1d_update`'s torch fallback.
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = _act(activation)(out)
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Prefill (parallel) form of the causal depthwise conv.

    Ported verbatim from `modeling_qwen3_5_moe.causal_conv1d_fn`'s torch
    fallback: left-pad by `kernel_size - 1` via the conv's own `padding`
    argument, then trim back to the input length.
    """
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = _act(activation)(out)
    return out.to(hidden_states.dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """L2-normalize, matching the FLA library's convention (rsqrt of sum of
    squares, not `F.normalize`'s slightly different epsilon placement)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Prefill-time gated delta rule, chunked along the sequence dimension.

    Ported verbatim (algorithm, variable names, and evaluation order kept
    identical on purpose so this is auditable line-by-line against the
    reference) from `modeling_qwen3_5_moe.torch_chunk_gated_delta_rule` in
    `transformers==5.16.1`. Runs the whole computation in float32 regardless
    of input dtype and casts back at the end -- required precision, not a
    stylistic choice: the recurrent state accumulates across the whole
    sequence and bf16 rounding there is visible in the output (see CLAUDE.md
    bug #13, a different accumulator-precision bug with the same shape).

    Args:
        query, key: `[batch, seq, num_k_heads, k_head_dim]`
        value: `[batch, seq, num_v_heads, v_head_dim]`
        g: log-space decay, `[batch, seq, num_v_heads]`, entries <= 0
        beta: per-token learning rate, `[batch, seq, num_v_heads]`
        initial_state: `[batch, num_v_heads, k_head_dim, v_head_dim]` or None
    Returns:
        (output `[batch, seq, num_v_heads, v_head_dim]`, final_state or None)
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Decode-time gated delta rule: one token at a time, state that never grows.

    This is the form that matters most for a streaming engine: the state is
    `[batch, num_v_heads, k_head_dim, v_head_dim]` regardless of how many
    tokens have been generated, so it can ride alongside a layer's evict/
    reload cycle exactly like a KV-cache entry does, rather than growing with
    context length like the chunked prefill path's intermediates do.

    Ported verbatim from `modeling_qwen3_5_moe.torch_recurrent_gated_delta_rule`
    in `transformers==5.16.1`. Same args/shapes as `chunk_gated_delta_rule`.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class RMSNormGated(nn.Module):
    """Gated RMSNorm: normalize, scale by weight, then gate with silu(z).

    Ported verbatim from `modeling_qwen3_5_moe.Qwen3_5MoeRMSNormGated`. Runs
    the norm and the gate activation in float32 regardless of input dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * _act(self.activation)(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention layer: causal short-conv + delta rule + gated norm.

    Parameter names and shapes match `Qwen3_5MoeGatedDeltaNet` exactly (
    `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `conv1d`, `dt_bias`,
    `A_log`, `norm.weight`, `out_proj.weight`), so a state dict trained/shipped
    against the reference module loads here unmodified, and this module can be
    swapped in for the reference one in a real `Qwen3_5MoeDecoderLayer`
    without touching any other layer's tensor names -- which is what makes it
    reachable by this project's existing shard/streaming machinery once
    `policy.py` and `shard.py` learn these tensor names (not yet done; see
    `model.py`).

    `config` is duck-typed: any object exposing `hidden_size`,
    `linear_num_value_heads`, `linear_num_key_heads`, `linear_key_head_dim`,
    `linear_value_head_dim`, `linear_conv_kernel_dim`, `hidden_act`,
    `rms_norm_eps`, and `layer_types` works -- the real
    `Qwen3_5MoeTextConfig`, or a lightweight stand-in for tests.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        # Lower bound kept away from 0 so log(A) never becomes -inf.
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        """Same call contract as `Qwen3_5MoeGatedDeltaNet.forward`.

        `cache_params`, if given, is anything implementing the
        `transformers.cache_utils.Cache` + `LinearAttentionLayer` protocol:
        `.has_previous_state(layer_idx, state_idx=0)`,
        `.layers[layer_idx].conv_states[0]` / `.recurrent_states[0]` /
        `.record_past`, `.update_conv_state(...)`, `.update_recurrent_state(...)`.
        We deliberately don't define our own cache class -- see the module
        docstring at the top of this file.
        """
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # `.float()` matters: without it, A can become -inf under fp16.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output
