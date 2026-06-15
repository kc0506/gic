# coding=utf-8
"""F0 pre-stress field as an inverse-map MLP (ported from gic_val b2).

The displacement is defined on the OBSERVED config x0 (rest UNKNOWN):
  w(x0) = MLP(x0)
  rest  X = x0 - w(x0)
  F0 = (I - grad_{x0} w)^{-1}     # compatible by construction (inverse map)

For a transverse bend (u_x = 0) grad w is nilpotent so (I - grad w)^{-1} = I + grad w
exactly.  The per-particle Jacobian is taken with torch.func (vmap+jacrev) so the MLP
weights stay in the graph; F0.backward(dL/dF0) -- with dL/dF0 read from the GIC BPTT
via estimator._read_F0_grad -- then reaches the weights.

NOTE: no `from __future__ import annotations` interactions here; pure torch.
"""
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap


class UMLP(nn.Module):
    """x (n,3) -> w (n,3) displacement.  Small tanh MLP, last layer ~0 so F0~=I at init
    (in-basin per the capture-radius lesson) but grads still flow to all layers."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 3),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(1e-3)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (n,3) -> (n,3)
        return self.net(x)


def F0_from_w(mlp: UMLP, x0: torch.Tensor, eye: torch.Tensor) -> tuple:
    """w=MLP(x0), F0=(I - grad_{x0} w)^{-1}.  Returns (F0 (N,3,3), Jw (N,3,3), w (N,3))."""
    params = dict(mlp.named_parameters())

    def w_of(p: torch.Tensor) -> torch.Tensor:        # (3,) -> (3,)
        return functional_call(mlp, params, (p.unsqueeze(0),)).squeeze(0)

    Jw = vmap(lambda p: jacrev(w_of)(p))(x0)          # (N,3,3) = grad_{x0} w
    w = vmap(w_of)(x0)                                  # (N,3)
    F0 = torch.linalg.inv(eye - Jw)                    # (N,3,3)
    return F0, Jw, w
