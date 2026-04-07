from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class QImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict[str, Any],
        obs_encoder: MultiImageObsEncoder,
        n_action_steps: int,
        n_obs_steps: int,
        value_hidden_dims: list[int],
        q_hidden_dims: list[int],
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1

        self.obs_encoder = obs_encoder
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_shape[0]
        self.action_chunk_dim = self.n_action_steps * self.action_dim
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.normalizer = LinearNormalizer()

        self.state_dim = self.obs_feature_dim * self.n_obs_steps
        self.value_head = _build_mlp(self.state_dim, value_hidden_dims, output_dim=1)
        self.q_head = _build_mlp(
            self.state_dim + self.action_chunk_dim,
            q_hidden_dims,
            output_dim=1,
        )

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("QImagePolicy is trained for value/Q estimation, not action prediction.")

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def value_parameters(self):
        return list(self.obs_encoder.parameters()) + list(self.value_head.parameters())

    def q_parameters(self):
        return self.q_head.parameters()

    def set_value_training_mode(self):
        self.obs_encoder.train()
        self.value_head.train()
        self.q_head.train()
        self.obs_encoder.requires_grad_(True)
        self.value_head.requires_grad_(True)
        self.q_head.requires_grad_(False)

    def set_q_training_mode(self):
        self.obs_encoder.eval()
        self.value_head.eval()
        self.q_head.train()
        self.obs_encoder.requires_grad_(False)
        self.value_head.requires_grad_(False)
        self.q_head.requires_grad_(True)

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = obs_dict if len(obs_dict) == 0 else self.normalizer.normalize(obs_dict)
        nobs = cast(Dict[str, torch.Tensor], nobs)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        obs_steps = min(value.shape[1], self.n_obs_steps)
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)
        obs_features = obs_features.reshape(batch_size, obs_steps, -1)
        return obs_features.reshape(batch_size, -1)

    def encode_state(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode_obs(obs_dict)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        naction = self.normalizer["action"].normalize(action)
        return naction.reshape(naction.shape[0], -1)

    def _extract_action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        if action.shape[1] < end:
            raise ValueError(
                f"Expected at least {end} action steps to extract the Q chunk, got {action.shape[1]}."
            )
        return action[:, start:end]

    def _extract_return_target(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "returns" in batch:
            returns = batch["returns"]
            return returns.reshape(-1, 1)

        returns_to_go = batch.get("returns_to_go")
        if returns_to_go is None:
            raise KeyError("Q/value training requires either 'returns' or 'returns_to_go' in the batch.")

        return_idx = self.n_obs_steps - 1
        if returns_to_go.shape[1] <= return_idx:
            raise ValueError(
                f"Expected returns_to_go to have at least {return_idx + 1} steps, got {returns_to_go.shape[1]}."
            )
        return returns_to_go[:, return_idx : return_idx + 1]

    def predict_value(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_head(self.encode_state(obs_dict))

    def predict_q(self, obs_dict: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        state = self.encode_state(obs_dict)
        action_features = self._normalize_action(action)
        return self.q_head(torch.cat([state, action_features], dim=1))

    def compute_value_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pred_values = self.predict_value(batch["obs"])
        target_returns = self._extract_return_target(batch)
        loss = F.mse_loss(pred_values, target_returns)
        return {
            "loss": loss,
            "value_loss": loss,
            "pred_value_mean": pred_values.mean(),
            "target_return_mean": target_returns.mean(),
        }

    def compute_q_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        state = self.encode_state(batch["obs"])
        action_chunk = self._extract_action_chunk(batch["action"])
        action_features = self._normalize_action(action_chunk)
        pred_advantages = self.q_head(torch.cat([state, action_features], dim=1))

        with torch.no_grad():
            current_values = self.value_head(state)
            target_returns = self._extract_return_target(batch)
            target_advantages = target_returns - current_values

        loss = F.mse_loss(pred_advantages, target_advantages)
        return {
            "loss": loss,
            "q_loss": loss,
            "pred_advantage_mean": pred_advantages.mean(),
            "target_advantage_mean": target_advantages.mean(),
        }
