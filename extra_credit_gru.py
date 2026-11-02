# Extra credit: a GRU cell written gate by gate (like LSTMScratch), checked against torch.nn.GRU.
from pathlib import Path

import torch
import torch.nn as nn


class GRUScratch(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.Wx = nn.Linear(in_dim, 3 * hidden)   # input weights for [reset, update, new], same order as nn.GRU
        self.Wh = nn.Linear(hidden, 3 * hidden)   # hidden weights (kept separate: r multiplies Wh h + b)

    def forward(self, x):                         # x: [batch, time, in_dim]
        h = x.new_zeros(x.shape[0], self.hidden)
        outputs = []
        for t in range(x.shape[1]):
            xr, xz, xn = self.Wx(x[:, t]).chunk(3, dim=-1)
            hr, hz, hn = self.Wh(h).chunk(3, dim=-1)
            r = torch.sigmoid(xr + hr)            # reset gate
            z = torch.sigmoid(xz + hz)            # update gate
            n = torch.tanh(xn + r * hn)           # new candidate state
            h = (1 - z) * n + z * h               # mix old and new state
            outputs.append(h)
        return torch.stack(outputs, dim=1)


def compare(dtype, steps=50):
    torch.manual_seed(0)
    ref = nn.GRU(16, 32, batch_first=True).to(dtype)
    mine = GRUScratch(16, 32).to(dtype)
    with torch.no_grad():                         # copy nn.GRU's weights into the scratch cell
        mine.Wx.weight.copy_(ref.weight_ih_l0)
        mine.Wx.bias.copy_(ref.bias_ih_l0)
        mine.Wh.weight.copy_(ref.weight_hh_l0)
        mine.Wh.bias.copy_(ref.bias_hh_l0)
    x = torch.randn(4, steps, 16, dtype=dtype)
    y_ref, y_mine = ref(x)[0], mine(x)
    y_ref.sum().backward()
    y_mine.sum().backward()
    out_diff = (y_ref - y_mine).abs().max().item()
    grad_diff = max((mine.Wx.weight.grad - ref.weight_ih_l0.grad).abs().max().item(),
                    (mine.Wh.weight.grad - ref.weight_hh_l0.grad).abs().max().item())
    return out_diff, grad_diff


lines = ["| precision | max output difference | max gradient difference |", "|---|---:|---:|"]
for dtype in (torch.float64, torch.float32):
    out_diff, grad_diff = compare(dtype)
    lines.append(f"| {str(dtype).split('.')[1]} | {out_diff:.1e} | {grad_diff:.1e} |")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
text = "## GRU written gate by gate vs. nn.GRU (50 time steps)\n\n" + "\n".join(lines) + "\n"
(OUT / "extra_credit_gru.md").write_text(text)
print(text)
