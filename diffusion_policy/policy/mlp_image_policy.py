from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

class MLPImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            aux_loss_weight: float = 0.0,
            loss_type: str = "nll",
            autoregressive: bool = True,
            **kwargs):
        assert loss_type in ("nll", "kl"), f"loss_type must be 'nll' or 'kl', got '{loss_type}'"
        
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        self.aux_loss_weight = aux_loss_weight
        self.loss_type = loss_type
        self.kwargs = kwargs
        self.autoregressive = autoregressive

        obs_input_dim = obs_feature_dim * n_obs_steps

        if autoregressive:
            input_dim = obs_input_dim + action_dim * n_action_steps
            self.action_output_dim = action_dim
        else:
            input_dim = obs_input_dim
            self.action_output_dim = n_action_steps * action_dim
        
        # Shared trunk
        layers = []
        last_dim = input_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last_dim, hidden_dim), nn.ReLU()]
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        
        # Separate heads for mean and log std
        self.mean_head = nn.Linear(last_dim, self.action_output_dim)
        self.log_std_head = nn.Linear(last_dim, self.action_output_dim)
        
        self.log_std_limits = (-5.0, 2.0)

        # Auxiliary reconstruction heads branch off encoder features (pre-trunk)
        self.aux_heads = nn.ModuleDict()
        auxiliary_shape_meta = shape_meta.get('auxiliary_obs', None)
        if auxiliary_shape_meta is not None and aux_loss_weight > 0:
            for key, attr in auxiliary_shape_meta.items():
                dim = attr['shape'][0]
                self.aux_heads[key] = nn.Sequential(
                    nn.Linear(obs_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                )

    def get_trunk_features(self, trunk_input: torch.Tensor) -> torch.Tensor:
        return self.trunk(trunk_input)

    def reshape_action_tensor(self, action_tensor: torch.Tensor) -> torch.Tensor:
        if self.autoregressive:
            return action_tensor
        return action_tensor.reshape(-1, self.n_action_steps, self.action_dim)

    def get_action_dist(self, h: torch.Tensor) -> Normal:
        mean = self.reshape_action_tensor(self.mean_head(h))
        log_std = self.reshape_action_tensor(self.log_std_head(h)).clamp(
            min=self.log_std_limits[0], max=self.log_std_limits[1]
        )
        return Normal(mean, torch.exp(log_std))

    def forward(self, trunk_input: torch.Tensor) -> Normal:
        return self.get_action_dist(self.get_trunk_features(trunk_input))

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize, encode, and flatten observations into (B, obs_feature_dim * n_obs_steps)."""
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1).reshape(B, -1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        obs_input = self._encode_obs(obs_dict)
        B = obs_input.shape[0]

        if self.autoregressive:
            Ta, Da = self.n_action_steps, self.action_dim
            action_slots = torch.zeros(B, Ta * Da, device=obs_input.device)
            predicted_actions = []
            for t in range(Ta):
                trunk_input = torch.cat([obs_input, action_slots], dim=-1)
                dist = self.forward(trunk_input)
                next_action = dist.mean
                predicted_actions.append(next_action)
                idx = t * Da
                action_slots = action_slots.clone()
                action_slots[:, idx:idx + Da] = next_action
            action_pred = torch.stack(predicted_actions, dim=1)
        else:
            dist = self.forward(obs_input)
            action_pred = dist.mean

        action = self.normalizer['action'].unnormalize(action_pred)
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps
        Ta = self.n_action_steps
        Da = self.action_dim
        
        # Encode obs
        if isinstance(nobs, dict):
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        else:
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, To, -1)
        obs_input = nobs_features.reshape(B, -1)

        start = To - 1
        end = start + Ta
        target = nactions[:, start:end]

        if self.autoregressive:
            # Teacher-forced: batch all steps in one forward pass.
            # For step t the trunk sees obs + [a_0, ..., a_{t-1}, 0, ..., 0].
            step_idx = torch.arange(Ta, device=target.device)
            mask = (step_idx.unsqueeze(1) > step_idx.unsqueeze(0)).float()
            prefixes = target.unsqueeze(1).expand(B, Ta, Ta, Da) * mask[None, :, :, None]
            action_slots = prefixes.reshape(B, Ta, Ta * Da)

            obs_expanded = obs_input.unsqueeze(1).expand(B, Ta, -1)
            trunk_input = torch.cat([obs_expanded, action_slots], dim=-1).reshape(B * Ta, -1)

            h = self.get_trunk_features(trunk_input)
            student_dist = self.get_action_dist(h)

            if self.loss_type == "kl":
                assert 'expert_dist' in batch, \
                    "loss_type='kl' requires expert distribution data in dataset"
                raw_mean = batch['expert_dist']['expert_action_mean'][:, start:end]
                raw_std = batch['expert_dist']['expert_action_std'][:, start:end]
                norm_expert_mean = self.normalizer['action'].normalize(raw_mean)
                action_scale = self.normalizer['action'].params_dict['scale']
                norm_expert_std = raw_std * action_scale
                expert_dist = Normal(
                    norm_expert_mean.reshape(B * Ta, Da),
                    norm_expert_std.reshape(B * Ta, Da),
                )
                per_step = torch.distributions.kl_divergence(
                    expert_dist, student_dist).sum(dim=-1)
            else:
                per_step = -student_dist.log_prob(target.reshape(B * Ta, Da)).sum(dim=-1)

            bc_loss = per_step.reshape(B, Ta).sum(dim=-1).mean()
        else:
            h = self.get_trunk_features(obs_input)
            student_dist = self.get_action_dist(h)

            if self.loss_type == "kl":
                assert 'expert_dist' in batch, \
                    "loss_type='kl' requires expert distribution data in dataset"
                raw_mean = batch['expert_dist']['expert_action_mean'][:, start:end]
                raw_std = batch['expert_dist']['expert_action_std'][:, start:end]
                norm_expert_mean = self.normalizer['action'].normalize(raw_mean)
                action_scale = self.normalizer['action'].params_dict['scale']
                norm_expert_std = raw_std * action_scale
                expert_dist = Normal(norm_expert_mean, norm_expert_std)
                bc_loss = torch.distributions.kl_divergence(
                    expert_dist, student_dist).sum(dim=(-1, -2)).mean()
            else:
                bc_loss = -student_dist.log_prob(target).sum(dim=(-1, -2)).mean()

        # Auxiliary reconstruction loss (from encoder features, not trunk)
        aux_loss = torch.tensor(0.0, device=obs_input.device)
        if self.aux_heads and 'auxiliary_obs' in batch:
            for key, head in self.aux_heads.items():
                pred = head(obs_input)
                aux_target = self.normalizer[key].normalize(
                    batch['auxiliary_obs'][key][:, To-1])
                aux_loss = aux_loss + F.mse_loss(pred, aux_target)

        loss = bc_loss + self.aux_loss_weight * aux_loss
        return {'loss': loss, 'bc_loss': bc_loss, 'aux_loss': aux_loss}