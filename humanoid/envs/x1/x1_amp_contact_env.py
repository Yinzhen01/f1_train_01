"""Contact-force tail only; all existing task and AMP behavior is inherited."""
import torch

from .x1_amp_refine_env import X1AMPRefineEnv
from humanoid.amp.contact import force_tail_cost


class X1AMPContactEnv(X1AMPRefineEnv):
    def reset_contact_diagnostics(self):
        self.contact_tail_calls = 0
        self.contact_tail_sum = torch.zeros((), device=self.device)
        self.contact_tail_positive = torch.zeros((), device=self.device, dtype=torch.long)

    def _reward_contact_force_tail(self):
        if not hasattr(self, 'contact_tail_calls'): self.reset_contact_diagnostics()
        value = force_tail_cost(self.contact_forces[:, self.feet_indices, :], self.cfg.rewards.max_contact_force)
        self.contact_tail_calls += 1
        self.contact_tail_sum += value.detach().sum()
        self.contact_tail_positive += (value.detach() > 0).sum()
        return value

    def contact_diagnostics(self):
        return dict(tail_calls=self.contact_tail_calls, tail_cost_sum=float(self.contact_tail_sum),
            tail_positive_env_steps=int(self.contact_tail_positive),
            runtime_scales={key: self.reward_scales.get(key, 0.) for key in
                            ('refine_slip', 'feet_contact_forces', 'contact_force_tail')})
