
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from .core import COH, NeuralModule

class ConstrainedRL:
    """Constrained Policy Optimization (simplified version)."""
    def __init__(self, coh: COH, policy_network: NeuralModule, gamma: float = 0.99,
                 constraint_cost: float = 1.0, lr: float = 1e-3):
        self.coh = coh
        self.policy = policy_network
        self.gamma = gamma
        self.constraint_cost = constraint_cost
        self.optimizer = optim.Adam(policy_network.module.parameters(), lr=lr)

    def collect_episode(self, max_steps: int = 100) -> Tuple[List[np.ndarray], List[int], List[float], List[float]]:
        """Run one episode, return (states, actions, returns, constraint_violations)."""
        states: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        constraint_violations: List[float] = []

        t = 0
        done = False
        while (not done) and t < max_steps:
            if self.coh.embedding is None:
                raise ValueError('COH must have an embedding function for RL.')
            state = self.coh.embedding(self.coh)
            states.append(state)
            with torch.no_grad():
                logits = self.policy.forward(torch.tensor(state, dtype=torch.float32))
                action_probs = torch.softmax(logits, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()
            action_name = list(self.coh.methods.keys())[action_idx]
            actions.append(action_idx)
            try:
                reward = float(self.coh.apply_method(action_name))
            except Exception:
                constraint_violations.append(1.0)
                reward = -float(self.constraint_cost)
            else:
                constraint_violations.append(0.0)
            rewards.append(reward)
            t += 1

        # Compute returns
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, float(G))
        return states, actions, returns, constraint_violations

    def train_episode(self, max_steps: int = 100) -> float:
        """Perform one training update using a collected episode."""
        states, actions, returns, _violations = self.collect_episode(max_steps)
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        logits = self.policy.forward(states_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        loss = - (action_log_probs * returns_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
