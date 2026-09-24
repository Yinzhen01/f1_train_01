"""Continuous LSGAN discriminator; independent gradients and frozen train stats."""
import torch
from torch import nn


class AMPDiscriminator(nn.Module):
    def __init__(self, spec, mean, std, hidden_dims=(256, 128), style_floor=0.):
        super().__init__()
        self.spec = spec
        if style_floor not in (0., -1.):
            raise ValueError("Unapproved style reward floor")
        self.style_floor = float(style_floor)
        if mean.shape != (spec.dim,) or std.shape != mean.shape or (std <= 0).any():
            raise ValueError("Invalid normalizer shape/scale")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not hidden_dims:
            raise ValueError("Invalid normalization/network config")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        layers, width = [], spec.history * spec.dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.append(nn.Linear(width, 1))  # Intentionally no Sigmoid.
        self.network = nn.Sequential(*layers)

    def normalized_flat(self, windows):
        if windows.ndim != 3 or windows.shape[1:] != (self.spec.history, self.spec.dim):
            raise ValueError("Expected [B,10,39] complete windows")
        if not torch.isfinite(windows).all():
            raise ValueError("NaN/Inf or padded window supplied to discriminator")
        return ((windows - self.mean) / self.std).reshape(len(windows), -1)

    def forward(self, windows):
        return self.network(self.normalized_flat(windows))

    def losses(self, demo, policy, gradient_penalty=10., bridge_gradient_penalty=0.):
        if not len(demo) or not len(policy) or gradient_penalty < 0 or bridge_gradient_penalty < 0:
            raise ValueError("Empty batch or invalid gradient penalty")
        demo_x = self.normalized_flat(demo.detach()).detach().requires_grad_(True)
        policy_x = self.normalized_flat(policy.detach()).detach()
        demo_score, policy_score = self.network(demo_x), self.network(policy_x)
        lsgan = .5 * ((demo_score - 1).square().mean() + (policy_score + 1).square().mean())
        grad = torch.autograd.grad(demo_score.sum(), demo_x, create_graph=True)[0]
        penalty = gradient_penalty * grad.square().sum(-1).mean()
        result = {"lsgan": lsgan, "gradient_penalty": penalty, "total": lsgan + penalty}
        if bridge_gradient_penalty:
            if demo_x.shape != policy_x.shape:
                raise ValueError("Bridge regularization requires equal batches")
            # Deterministic stratified coefficients: no global RNG consumption.
            # Mixtures are only gradient-penalty probes, never labelled demos.
            alpha = torch.linspace(.05, .95, len(demo_x), device=demo_x.device, dtype=demo_x.dtype)[:, None]
            mixed = (alpha*demo_x.detach()+(1-alpha)*policy_x).detach().requires_grad_(True)
            mixed_grad = torch.autograd.grad(self.network(mixed).sum(), mixed, create_graph=True)[0]
            bridge_penalty = bridge_gradient_penalty*mixed_grad.square().sum(-1).mean()
            result["bridge_gradient_penalty"] = bridge_penalty
            result["total"] = result["total"]+bridge_penalty
        return result

    @torch.no_grad()
    def style_reward(self, windows, scale=1., dt=.01):
        if scale < 0 or abs(dt - 1 / self.spec.fps) > 1e-10:
            raise ValueError("Invalid style scale/dt")
        score = self(windows).squeeze(-1)
        return dt * scale * (1 - .25 * (score - 1).square()).clamp_min(self.style_floor)


class DiscriminatorTrainer:
    def __init__(self, discriminator, learning_rate=1e-4, gradient_penalty=10., max_grad_norm=1., bridge_gradient_penalty=0.):
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        self.gradient_penalty, self.max_grad_norm = gradient_penalty, max_grad_norm
        self.bridge_gradient_penalty = bridge_gradient_penalty

    def step(self, demo, policy):
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.discriminator.losses(demo, policy, self.gradient_penalty, self.bridge_gradient_penalty)
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise ValueError("Nonfinite discriminator loss")
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}
