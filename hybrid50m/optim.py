"""Optimizer + learning-rate schedules.

`MuonAdamW`: Muon (orthogonalised momentum) for the 2-D matrices inside the transformer blocks, AdamW for everything else
(embeddings, projections, norms). One torch Optimizer, so GradScaler / state_dict / checkpoints treat it like any other.
Muon's update is rescaled to the RMS of an AdamW update (Moonlight, 0.2*sqrt(max(shape))), so ONE learning rate and ONE
weight decay serve both halves. `muon=False` gives plain AdamW with the same parameter groups.
"""
import math

import torch


def newton_schulz(G, steps=5):
    # quintic iteration that pushes all singular values of G towards 1 (the "orthogonalised" update); fp32 on purpose
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    flip = X.size(0) > X.size(1)
    if flip:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if flip else X


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, groups, lr, betas=(0.9, 0.95), eps=1e-8, momentum=0.95, ns_steps=5):
        # groups: list of dicts {"params": [...], "kind": "muon" | "adamw", "weight_decay": float}
        defaults = dict(lr=lr, betas=betas, eps=eps, momentum=momentum, ns_steps=ns_steps, weight_decay=0.0, kind="adamw")
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g["kind"] == "muon":
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p)
                    st["buf"].mul_(g["momentum"]).add_(p.grad)
                    upd = newton_schulz(p.grad + g["momentum"] * st["buf"], g["ns_steps"])   # Nesterov momentum
                    p.mul_(1 - lr * wd)
                    p.add_(upd.to(p.dtype), alpha=-lr * 0.2 * math.sqrt(max(p.shape)))
                else:
                    if not st:
                        st["t"] = 0
                        st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    st["t"] += 1
                    b1, b2 = g["betas"]
                    st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (st["v"] / (1 - b2 ** st["t"])).sqrt_().add_(g["eps"])
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(st["m"], denom, value=-lr / (1 - b1 ** st["t"]))


def build_optimizer(model, lr, weight_decay=0.1, muon=True):
    muon_p, decay_p, plain_p = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if muon and p.dim() == 2 and name.startswith("blocks."):
            muon_p.append(p)
        elif p.dim() >= 2:
            decay_p.append(p)
        else:
            plain_p.append(p)
    groups = [{"params": decay_p, "kind": "adamw", "weight_decay": weight_decay},
              {"params": plain_p, "kind": "adamw", "weight_decay": 0.0}]
    if muon_p:
        groups.insert(0, {"params": muon_p, "kind": "muon", "weight_decay": weight_decay})
    return MuonAdamW(groups, lr=lr)


def lr_at(step, total, base, warmup, kind="wsd", decay_frac=0.2, min_ratio=0.02):
    """Learning rate for optimizer step `step` (0-based).

    wsd      warmup -> flat -> last `decay_frac` of the run decays as 1 - sqrt(progress). The flat part does not depend on
             `total`, so a run can be extended (or stopped) without redoing the schedule, as long as it has not started decaying.
    cosine   warmup -> cosine to min_ratio * base
    constant warmup -> flat (used by the pilot, where arms are compared at equal wall-clock time)
    """
    if step < warmup:
        return base * (step + 1) / warmup
    if kind == "constant":
        return base
    if kind == "cosine":
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
    start = int(total * (1 - decay_frac))
    if step < start:
        return base
    p = min(1.0, (step - start) / max(1, total - start))
    return base * (min_ratio + (1 - min_ratio) * (1 - math.sqrt(p)))
