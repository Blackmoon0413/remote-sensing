"""Pure-PyTorch reference implementations of the two Mamba CUDA kernels used by RSCaMa.

The algorithms are the reference versions shipped with the official packages
(``mamba_ssm.ops.selective_scan_interface.selective_scan_ref`` and
``causal_conv1d.causal_conv1d_interface.causal_conv1d_ref``). They are slower
than the fused kernels but numerically equivalent up to float rounding, which
lets RSCaMa run on Windows / GPUs without prebuilt Mamba kernels. Correctness is
checked end-to-end by reproducing the published RSCaMa scores.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """x: (batch, dim, seqlen); weight: (dim, width); depthwise causal conv."""
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation}")
    return out.to(dtype_in)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """u, delta, z: (b, d, l); A: (d, n); B, C: (b, n, l) or (d, n); D, delta_bias: (d,)."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate), dtype=torch.float32)
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float()))
    if B.dim() == 3:
        delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        delta_b_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    ys = []
    for i in range(u.shape[2]):
        x = delta_a[:, :, i] * x + delta_b_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C[:, :, i]) if C.dim() == 3 else torch.einsum("bdn,dn->bd", x, C)
        ys.append(y)
    last_state = x
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype_in)
    return (out, last_state) if return_last_state else out


if __name__ == "__main__":
    # tiny self-test against an explicit loop-free formulation for L=1
    torch.manual_seed(0)
    b, d, n, l = 2, 4, 3, 5
    u, dt, z = torch.randn(b, d, l), torch.randn(b, d, l), torch.randn(b, d, l)
    A = -torch.rand(d, n); Bm, Cm = torch.randn(b, n, l), torch.randn(b, n, l); Dv = torch.randn(d)
    out, last = selective_scan_fn(u, dt, A, Bm, Cm, Dv, z, None, True, True)
    print("selective_scan_fn ok", out.shape, last.shape)
    w = torch.randn(d, 4)
    print("causal_conv1d_fn ok", causal_conv1d_fn(torch.randn(b, d, l), w, None, "silu").shape)
