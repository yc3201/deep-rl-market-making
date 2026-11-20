"""PPO (from scratch) for the market-making agent.

The policy is the paper's architecture: the PRETRAINED Attn-LOB
encoder as a frozen feature trunk, plus small actor/critic heads on
[encoder features (64) + agent state (4)]. The paper fine-tunes the
pretrained network as the function approximator during RL; freezing
the trunk here is a deliberate CPU-time choice (the encoder forward
per step is the bottleneck, and a frozen trunk lets one batched
forward per episode precompute all market features). The heads are
trained; unfreezing the trunk is a listed extension (see README).

Action: 2-dim continuous (a1, a2) in [0, 1]^2. The actor outputs a
Gaussian mean through a sigmoid (so the mean stays in [0, 1]) and a
learnable per-dimension log-std; sampled actions are clipped to
[0, 1]. Log-probs are those of the pre-clip Gaussian (standard
PPO-with-clipping practice; the clip rate is logged).

Rewards are scaled by 1 / market spread (i.e. expressed in spread
units, like the paper's ND-PnL) so the value function sees O(1)
targets. Advantages are normalised per update batch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


def mlp(in_dim: int, hidden: int = 64, out_dim: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, encoder, feat_dim: int = 64, state_dim: int = 4,
                 hidden: int = 64, init_std: float = 0.3,
                 init_a1: float = 0.5, init_a2: float = 0.5):
        """init_a1 / init_a2: the initial means of the skew / spread
        actions (0..1).

        Warm start (init_a2 ~ 0.9): the actor's final bias is set so
        the policy starts by quoting wide (near the calibrated
        fixed-spread baseline) and PPO refines from there. This is
        needed because from-scratch PPO never learns wide quoting in
        this env: the smoke run and the V1 run (reward fixed, floor,
        more entropy) both ended with a2 ~ 0.5 constant and 4-6
        fills/ep, i.e. the policy never discovers that wide quoting
        wins — the advantage signal for width is drowned by per-step
        mid-price noise and the updates (clip_frac 0.3-0.5) make the
        policy wander instead of climb. Initializing at the calibrated
        optimum and refining is the standard warm-start practice;
        it is disclosed in the README, and the eval still has to earn
        any edge over fixed by state-dependent quoting.
        """
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.actor = mlp(feat_dim + state_dim, hidden, 2)
        self.critic = mlp(feat_dim + state_dim, hidden, 1)
        # Warm-start the actor's output bias: a1 mean init_a1 (no skew
        # when ~0), a2 mean init_a2 (wide when ~0.9). The random weights
        # still let features modulate the action around the mean.
        with torch.no_grad():
            b = self.actor[-1].bias
            ia1 = min(max(init_a1, 0.01), 0.99)
            b[0] = float(np.log(ia1 / (1.0 - ia1)))
            ia = min(max(init_a2, 0.01), 0.99)
            b[1] = float(np.log(ia / (1.0 - ia)))
        self.log_std = nn.Parameter(
            torch.full((2,), float(np.log(init_std))))

    def _features(self, lob: torch.Tensor,
                  agent_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.encoder.forward_features(lob)
        return torch.cat([f, agent_state], dim=-1)

    def act_from_features(self, x):
        """Sample an action from precomputed 68-dim features."""
        mean = torch.sigmoid(self.actor(x))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        a = dist.sample()
        logprob = dist.log_prob(a).sum(-1)
        return a.clamp(0, 1), logprob, self.critic(x).squeeze(-1)

    def act(self, lob, agent_state):
        """Sample an action from a raw LOB window + agent state."""
        return self.act_from_features(self._features(lob, agent_state))

    def act_deterministic(self, lob, agent_state):
        x = self._features(lob, agent_state)
        return torch.sigmoid(self.actor(x))

    def act_deterministic_from_features(self, x):
        return torch.sigmoid(self.actor(x))


def episode_features(encoder, feats, mid, start, episode_len, window,
                     device) -> torch.Tensor:
    """One batched encoder forward for a whole episode: [L, feat_dim].

    Window for step t ends at global index start + t. (Market features
    depend only on the LOB, not on the agent, so they are precomputed.)
    """
    idx = np.arange(start, start + episode_len)
    wins = np.stack([feats[:, i - window + 1:i + 1].T
                     for i in idx]).astype(np.float32)
    with torch.no_grad():
        f = encoder.forward_features(torch.from_numpy(wins).to(device))
    return f.cpu()


def collect_episode(env, policy, feats_t, device, reward_scale: float):
    """Roll out one episode with the stochastic policy.

    Returns dict of per-step arrays (length L): features [L, 68],
    actions, logprobs, rewards (scaled), values, dones, plus the raw
    episode PnL (z-units) and fills.
    """
    obs = env.reset()
    L = env.p.episode_len
    agent_states = np.zeros((L, 4), dtype=np.float32)
    actions = np.zeros((L, 2), dtype=np.float32)
    logprobs = np.zeros(L, dtype=np.float32)
    rewards = np.zeros(L, dtype=np.float32)
    values = np.zeros(L, dtype=np.float32)
    for t in range(L):
        # Market features are precomputed for the whole episode; only
        # the agent state (inventory, time) is live.
        x = torch.from_numpy(
            np.concatenate([feats_t[t].numpy(), obs["agent_state"]])[None]
        ).to(device)
        with torch.no_grad():
            a, lp, v = policy.act_from_features(x)
        agent_states[t] = obs["agent_state"]
        actions[t] = a.cpu().numpy()[0]
        logprobs[t] = lp.cpu().numpy()[0]
        values[t] = v.cpu().numpy()[0]
        obs, r, done, _ = env.step(tuple(actions[t]))
        rewards[t] = r * reward_scale
        if done:
            break
    return {
        "features": np.concatenate(
            [feats_t.numpy(), agent_states], axis=1).astype(np.float32),
        "actions": actions, "logprobs": logprobs, "rewards": rewards,
        "values": values, "pnl": env.pnl, "fills": env.fills,
        "mean_abs_inv": float(np.mean(np.abs(env.inv_curve))),
    }


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Generalised advantage estimation; episode ends -> next value 0."""
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def ppo_update(policy, opt, buf, epochs=4, minibatch=256, clip=0.2,
               ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cpu"):
    """One PPO update over the collected buffer (dict of arrays)."""
    feats = torch.from_numpy(buf["features"]).to(device)
    acts = torch.from_numpy(buf["actions"]).to(device)
    old_lp = torch.from_numpy(buf["logprobs"]).to(device)
    adv = torch.from_numpy(buf["advantages"]).to(device)
    rets = torch.from_numpy(buf["returns"]).to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(feats)
    stats = {"clip_frac": [], "entropy": [], "pg_loss": [], "vf_loss": []}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            # The buffer stores precomputed 68-dim features (frozen
            # encoder trunk), so the update applies the heads directly.
            lp, val, ent = _evaluate_from_features(policy, feats[idx],
                                                   acts[idx])
            ratio = torch.exp(lp - old_lp[idx])
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]
            pg_loss = -torch.min(surr1, surr2).mean()
            vf_loss = ((val - rets[idx]) ** 2).mean()
            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()
            stats["clip_frac"].append(
                (torch.abs(ratio - 1) > clip).float().mean().item())
            stats["entropy"].append(ent.mean().item())
            stats["pg_loss"].append(pg_loss.item())
            stats["vf_loss"].append(vf_loss.item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def _evaluate_from_features(policy, feats68, action):
    """Logprob/value/entropy given precomputed 68-dim features.

    The encoder trunk is frozen, so during the update the features are
    fixed inputs: the actor/critic heads are applied directly.
    """
    mean = torch.sigmoid(policy.actor(feats68))
    std = policy.log_std.exp().expand_as(mean)
    dist = torch.distributions.Normal(mean, std)
    logprob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
    return logprob, policy.critic(feats68).squeeze(-1), entropy
