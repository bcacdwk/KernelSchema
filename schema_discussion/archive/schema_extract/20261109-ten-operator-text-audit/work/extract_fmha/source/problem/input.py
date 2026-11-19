import math
import torch

def joint_qkv(nq, nk, heads, kv_heads, dim, value_dim, device):
    """THD operands. Common latent topics induce QK affinity and KV structure.

    Four latent topics and eight fixed sinusoidal projection components are
    synthetic assumptions. They are not a reproduction of learned weights.
    """
    assert heads % kv_heads == 0
    position = torch.arange(max(dim, value_dim), device=device).float()
    basis = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * position[None, :] + 0.3) / 17.0)
    basis = basis / basis.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    latent = torch.randn((kv_heads, 4, 8), device=device)
    centers = latent @ basis[:, :dim]
    vcenters = latent @ torch.cos((torch.arange(1, 9, device=device).float()[:, None] * position[None, :value_dim] + 0.7) / 19.0) / 2.0
    k = torch.empty((nk, kv_heads, dim), device=device)
    v = torch.empty((nk, kv_heads, value_dim), device=device)
    kgain = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    cgain = torch.linspace(0.7, 1.3, dim, device=device)
    hidx = torch.arange(kv_heads, device=device)[None, :]
    for start in range(0, nk, 1024):
        count = min(1024, nk - start)
        topic = torch.randint(4, (count, kv_heads), device=device)
        k[start:start + count] = (0.8 * torch.randn((count, kv_heads, dim), device=device) + 0.6 * centers[hidx, topic]) * kgain * cgain
        v[start:start + count] = (0.8 * torch.randn((count, kv_heads, value_dim), device=device) + 0.6 * vcenters[hidx, topic]) * kgain
    q = torch.empty((nq, heads, dim), device=device)
    qcenters = centers.repeat_interleave(heads // kv_heads, 0)
    qhead = torch.arange(heads, device=device)[None, :]
    qgain = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for start in range(0, nq, 1024):
        count = min(1024, nq - start)
        topic = torch.randint(4, (count, heads), device=device)
        q[start:start + count] = (0.8 * torch.randn((count, heads, dim), device=device) + 0.6 * qcenters[qhead, topic]) * qgain * cgain
    if dim == 576:
        q[..., 512:] *= 0.5
        k[..., 512:] *= 0.5
    return (q, k, v)
FP8_MAX_NORM = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def e8m0_ceil(v: torch.Tensor) -> torch.Tensor:
    """fp32 tensor (>= 0) -> biased E8M0 exponent byte, TE ``float_to_e8m0`` semantics.

    Rounds the scale UP to the next power of two (unless the value already is
    one).  0 -> 0x00, inf -> 0xFE (satfinite), nan -> 0xFF.
    """
    v = v.float().contiguous()
    bits = v.view(torch.int32)
    exp = bits >> 23 & 255
    mant = bits & 8388607
    round_up = (mant > 0) & (exp != 254) & ~((exp == 0) & (mant <= 4194304))
    e = exp + round_up.to(torch.int32)
    e = torch.where(torch.isnan(v), torch.full_like(e, 255), e)
    e = torch.where(torch.isinf(v), torch.full_like(e, 254), e)
    return e.to(torch.uint8)

def exp2_rcp(e: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 byte -> exact fp32 ``2^(127-e)`` (TE ``exp2f_rcp``), incl. the
    ``e == 254 -> 2^-127`` (fp32 subnormal 0x00400000) special case."""
    e32 = e.to(torch.int32)
    rcp_bits = torch.where(e32 == 254, torch.full_like(e32, 4194304), 254 - e32 << 23)
    return rcp_bits.view(torch.float32)

def quantize_blocks(x: torch.Tensor, fp8_dtype: torch.dtype):
    """Quantize ``x[..., 32]`` blocks (last dim = one 32-element block).

    Returns ``(data fp8 same shape, e8m0 uint8 x.shape[:-1])``.
    """
    max_norm = FP8_MAX_NORM[fp8_dtype]
    x = x.float()
    amax = x.abs().amax(dim=-1)
    inv = torch.tensor(1.0 / max_norm, dtype=torch.float32, device=x.device)
    e = e8m0_ceil(amax * inv)
    rcp = exp2_rcp(e).unsqueeze(-1)
    data = (x * rcp).clamp_(-max_norm, max_norm).to(fp8_dtype)
    return (data, e)

def quantize_mxfp8_2d(t2d: torch.Tensor, fp8_dtype: torch.dtype):
    """TE ``MXFP8Quantizer(rowwise=True, columnwise=True)`` on a 2-D ``[M, K]``
    tensor with M, K multiples of 128 (SDPA tests pre-pad to this).

    Returns ``(rowwise_data, rowwise_scale_inv, columnwise_data,
    columnwise_scale_inv)`` with TE's shapes:
    rowwise_data fp8 ``[M, K]`` (scaled along K), rowwise_scale_inv uint8
    ``[M, K//32]``; columnwise_data fp8 ``[M, K]`` (scaled along M, NOT
    transposed), columnwise_scale_inv uint8 ``[M//32, K]``.
    """
    M, K = t2d.shape
    assert M % 128 == 0 and K % 128 == 0, f'pre-pad to 128 multiples, got {t2d.shape}'
    x = t2d.float()
    row_data, row_e = quantize_blocks(x.reshape(M, K // 32, 32), fp8_dtype)
    col_data_t, col_e_t = quantize_blocks(x.T.contiguous().reshape(K, M // 32, 32), fp8_dtype)
    return (row_data.reshape(M, K), row_e, col_data_t.reshape(K, M).T.contiguous(), col_e_t.transpose(0, 1).contiguous())

def swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """Core F8_128x4 reorder of a logical scale matrix ``[..., R, C]`` (uint8),
    R % 128 == 0, C % 4 == 0.  Atoms (128 rows x 4 cols = 512 B) are laid out
    row-major; inside an atom, scale (r, c) lives at byte
    ``(r % 32) * 16 + (r // 32) * 4 + c``.  Output is uint8 ``[..., R, C]``
    (same byte count; only the physical order matters)."""
    assert sf.dtype == torch.uint8
    *lead, R, C = sf.shape
    assert R % 128 == 0 and C % 4 == 0, f'F8_128x4 needs R%128==0, C%4==0, got {sf.shape}'
    v = sf.reshape(*lead, R // 128, 4, 32, C // 4, 4)
    n = len(lead)
    v = v.permute(*range(n), n + 0, n + 3, n + 2, n + 1, n + 4)
    return v.contiguous().reshape(*lead, R, C)

def swizzle_sf_rowwise(sf: torch.Tensor) -> torch.Tensor:
    """Swizzle a rowwise scale_inv ``[..., M, K//32]`` (scales along the
    contraction dim K) into cuDNN's F8_128x4 order (TE
    ``swizzle_row_scaling_kernel``)."""
    return swizzle_128x4(sf)

def swizzle_sf_columnwise(sf: torch.Tensor) -> torch.Tensor:
    """Swizzle a columnwise scale_inv stored as ``[..., M//32, K]`` (TE storage;
    scales along M) into cuDNN's F8_128x4 order for the transposed operand
    ``[K, M]`` (TE ``swizzle_col_scaling_kernel``): the 128x4 rule is applied to
    the transposed scale matrix ``[K, M//32]``.  Output keeps the storage shape
    ``[..., M//32, K]`` (bytes are what the kernel consumes)."""
    out = swizzle_128x4(sf.transpose(-1, -2).contiguous())
    return out.reshape(sf.shape)

def quantize_to_mxfp8(tensor, b, h, s, d, block_size=32, fp8_dtype=torch.float8_e4m3fn):
    """Torch-only drop-in for the TE-based ``quantize_to_mxfp8`` in
    ``test/python/sdpa/mxfp8.py`` (same signature and return convention).

    Returns the rowwise and columnwise payloads with their F8_128x4 scale
    tensors. The reference reconstructs values directly from stored scales.
    """
    assert block_size == 32, 'MXFP8 uses 32-element blocks'
    l = b * h
    d_scale_padded = ceil_div(ceil_div(d, block_size), 4) * 4
    d_padded = d_scale_padded * block_size
    s_scale_padded = ceil_div(ceil_div(s, block_size), 4) * 4
    s_padded = s_scale_padded * block_size
    tensor_3d = tensor.float().reshape(l, s, d)
    pad_d = d_padded - d
    pad_s = s_padded - s
    if pad_s > 0 or pad_d > 0:
        tensor_3d = torch.nn.functional.pad(tensor_3d, (0, pad_d, 0, pad_s))
    tensor_2d = tensor_3d.reshape(l * s_padded, d_padded)
    row_data, row_e, col_data, col_e = quantize_mxfp8_2d(tensor_2d, fp8_dtype)
    fp8_data_d = row_data.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous().reshape(b, h, s, d)
    sf_d_swizzle = swizzle_sf_rowwise(row_e)
    fp8_data_s = col_data.reshape(l, s_padded, d_padded)[:, :s, :d].contiguous().reshape(b, h, s, d)
    sf_s_swizzle = swizzle_sf_columnwise(col_e)
    return (fp8_data_d, None, sf_d_swizzle, fp8_data_s, None, sf_s_swizzle)

def guarded(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return torch.full((n + 256,), -17, dtype=dtype, device='cuda')[128:-128].view(shape)

def make_payload(batch, heads, kv_heads, query_len, kv_len, head_dim, causal, window_left=None, distribution='normal', return_lse=True):
    rq, rk, rv = joint_qkv(batch * query_len, batch * kv_len, heads, kv_heads, head_dim, head_dim, 'cuda')

    def operand(raw, h, s, columnwise=False):
        t = raw.reshape(batch, s, h, head_dim).transpose(1, 2).to(torch.bfloat16)
        if distribution == 'tiny':
            t *= 0.01
        if distribution == 'peaked':
            t *= 3
        if distribution == 'zero':
            t.zero_()
        rows = quantize_to_mxfp8(t, batch, h, s, head_dim)
        data, sf = (rows[3], rows[5]) if columnwise else (rows[0], rows[2])
        return (data.transpose(1, 2).contiguous().transpose(1, 2), sf)
    q, sq = operand(rq, heads, query_len)
    k, sk = operand(rk, kv_heads, kv_len)
    v, sv = operand(rv, kv_heads, kv_len, True)
    out = guarded((batch, query_len, heads, head_dim), torch.bfloat16).transpose(1, 2)
    lse = guarded((batch, heads, query_len), torch.float32) if return_lse else None
    scale = head_dim ** (-0.5)
    call = dict(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=out, lse_tensor=lse, sf_q=sq, sf_k=sk, sf_v=sv)
    if not causal:
        call['seq_kv_lens'] = torch.full((batch,), kv_len, device='cuda', dtype=torch.int32)
    return call

def _make_inputs(batch, head_dim, heads, kv_heads, kv_len, query_len, causal=None, distribution='normal', return_lse=True, window_left=None, device='cuda'):
    shape = {'device': device, 'distribution': distribution, 'return_lse': return_lse, **{k: v for k, v in {'batch': batch, 'causal': causal, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_len': kv_len, 'query_len': query_len, 'window_left': window_left}.items() if v is not None}}
    if device != 'cuda':
        raise ValueError('Native input generator requires CUDA')
    shape.pop('device')
    p = make_payload(**shape)
    return {'q_tensor':p['q_tensor'], 'k_tensor':p['k_tensor'], 'v_tensor':p['v_tensor'], 'o_tensor':p['o_tensor'], 'lse_tensor':p.get('lse_tensor'), 'sinks':p.get('sinks'), 'seq_q_lens':p.get('seq_q_lens'), 'seq_kv_lens':p.get('seq_kv_lens'), 'scale_softmax':p.get('scale_softmax'), 'current_stream':p.get('current_stream'), 'sf_q':p.get('sf_q'), 'sf_k':p.get('sf_k'), 'sf_v':p.get('sf_v'), 'amax_o':p.get('amax_o'), 'descale_q':p.get('descale_q'), 'descale_k':p.get('descale_k'), 'descale_v':p.get('descale_v'), 'scale_o':p.get('scale_o'), 'workspace':p.get('workspace'), 'block_table':p.get('block_table'), 'block_table_v':p.get('block_table_v'), 'gate':p.get('gate'), 'sf_o':p.get('sf_o')}

def prepare_model(model_class, inputs, init_kwargs, input_kwargs):
    """Keep cuDNN graph preparation and its required workspace outside timing."""
    g = input_kwargs
    # The public -1 sentinel aliases no window; the native constructor uses None.
    window_left = g.get('window_left')
    if window_left == -1:
        window_left = None
    model = model_class(sample_q=inputs['q_tensor'], sample_k=inputs['k_tensor'],
        sample_v=inputs['v_tensor'], sample_o=inputs['o_tensor'], sample_lse=inputs['lse_tensor'],
        is_causal=g['causal'], causal_bottom_right=g['causal'], window_size_left=window_left,
        scale_softmax=g['head_dim'] ** -0.5, dtype_o=torch.bfloat16, pertensor_fp8=False,
        has_amax_o=False, seq_kv_lens_present=not g['causal'])
    assert model.check_support()
    model.compile()
    inputs['workspace'] = torch.empty(model.scratch_workspace_bytes(), dtype=torch.uint8, device=inputs['q_tensor'].device)
    return model, inputs

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['o_tensor', 'lse_tensor', 'workspace']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['o_tensor', 'lse_tensor']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
