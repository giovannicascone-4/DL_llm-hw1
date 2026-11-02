"""Extra credit: GRU cell written gate by gate, checked against nn.GRU (outputs and gradients)."""
from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout

import torch
import torch.nn as nn


class GRUScratch(nn.Module):
    """A single-layer GRU, one time step at a time (for reading, not for speed)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.Wx = nn.Linear(in_dim, 3 * hidden)   # rows: [r | z | n], same order as nn.GRU
        self.Wh = nn.Linear(hidden, 3 * hidden)

    def cell(self, x_t, h):
        xr, xz, xn = self.Wx(x_t).chunk(3, dim=-1)
        hr, hz, hn = self.Wh(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)                # reset gate: how much of the past feeds the candidate
        z = torch.sigmoid(xz + hz)                # update gate: keep old state (z->1) or overwrite (z->0)
        n = torch.tanh(xn + r * hn)               # candidate state
        return (1 - z) * n + z * h                # convex mix -> an additive, gradient-friendly path

    def forward(self, x, h0=None):                # x [B, T, D] -> outputs [B, T, H], h_T [B, H]
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden) if h0 is None else h0
        outs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, dim=1), h

    @torch.no_grad()
    def load_from_torch(self, gru: nn.GRU, layer: int = 0):
        self.Wx.weight.copy_(getattr(gru, f"weight_ih_l{layer}"))
        self.Wx.bias.copy_(getattr(gru, f"bias_ih_l{layer}"))
        self.Wh.weight.copy_(getattr(gru, f"weight_hh_l{layer}"))
        self.Wh.bias.copy_(getattr(gru, f"bias_hh_l{layer}"))


class StackedGRUScratch(nn.Module):
    def __init__(self, in_dim, hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([GRUScratch(in_dim if i == 0 else hidden, hidden) for i in range(n_layers)])

    def forward(self, x, h0=None):
        hs = []
        for i, layer in enumerate(self.layers):
            x, h = layer(x, None if h0 is None else h0[i])
            hs.append(h)
        return x, torch.stack(hs)


def check(B=4, T=37, D=24, H=32, n_layers=2, seed=0, with_h0=True):
    torch.manual_seed(seed)
    ref = nn.GRU(D, H, num_layers=n_layers, batch_first=True).double()
    mine = StackedGRUScratch(D, H, n_layers).double()
    for i, layer in enumerate(mine.layers):
        layer.load_from_torch(ref, i)
    x = torch.randn(B, T, D, dtype=torch.float64)
    h0 = torch.randn(n_layers, B, H, dtype=torch.float64) if with_h0 else None
    y_ref, h_ref = ref(x, h0)
    y_me, h_me = mine(x, h0)
    out_err = (y_ref - y_me).abs().max().item()
    h_err = (h_ref - h_me).abs().max().item()
    # gradients: same random upstream gradient into both
    gy = torch.randn_like(y_ref)
    (y_ref * gy).sum().backward()
    (y_me * gy).sum().backward()
    g_err = 0.0
    for i, layer in enumerate(mine.layers):
        for mine_p, name in [(layer.Wx.weight, "weight_ih"), (layer.Wx.bias, "bias_ih"),
                             (layer.Wh.weight, "weight_hh"), (layer.Wh.bias, "bias_hh")]:
            g_err = max(g_err, (mine_p.grad - getattr(ref, f"{name}_l{i}").grad).abs().max().item())
    return out_err, h_err, g_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("GRUScratch vs torch.nn.GRU (float64, weights copied from nn.GRU)\n")
        print(f"{'case':<52s} {'max|dy|':>10s} {'max|dh_T|':>10s} {'max|dgrad|':>11s}  ok")
        cases = [dict(n_layers=1, with_h0=False), dict(n_layers=1, with_h0=True), dict(n_layers=2, with_h0=True),
                 dict(n_layers=3, T=128, H=64, with_h0=True), dict(n_layers=2, B=1, T=1, with_h0=False)]
        all_ok = True
        for c in cases:
            o, h, g = check(**c)
            ok = max(o, h, g) < 1e-10
            all_ok &= ok
            print(f"{str(c):<52s} {o:10.2e} {h:10.2e} {g:11.2e}  {'PASS' if ok else 'FAIL'}")
        # float32 sanity (what training would use): tolerance ~1e-5
        torch.manual_seed(1)
        ref = nn.GRU(16, 16, batch_first=True)
        me = GRUScratch(16, 16)
        me.load_from_torch(ref)
        x = torch.randn(2, 50, 16)
        e32 = (ref(x)[0] - me(x)[0]).abs().max().item()
        print(f"\nfloat32, T=50: max|dy| = {e32:.2e} ({'PASS' if e32 < 1e-5 else 'FAIL'})")
        print(f"\nOVERALL: {'PASS' if all_ok and e32 < 1e-5 else 'FAIL'}")
    txt = buf.getvalue()
    print(txt)
    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
