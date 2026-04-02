from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

class ActionChunkTransformerPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            aux_loss_weight: float = 0.0,
            loss_type: str = "nll",
            dropout: float = 0.1,
            horizon: int = 12,
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
        self.hidden_dim = hidden_dim

        aux_input_dim = obs_feature_dim * n_obs_steps
        self.max_seq_len = n_obs_steps + 1 + n_action_steps
        
        # transformer model
        config = GPT2Config(
            n_positions=self.max_seq_len,
            n_embd=hidden_dim,
            n_layer=hidden_depth,
            n_head=8,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.obs_proj = nn.Linear(obs_feature_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Separate heads for mean and log std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_limits = (-5.0, 2.0)

        # Auxiliary reconstruction heads branch off encoder features (pre-trunk)
        # to directly pressure the visual encoder to retain state information
        self.aux_heads = nn.ModuleDict()
        auxiliary_shape_meta = shape_meta.get('auxiliary_obs', None)
        if auxiliary_shape_meta is not None and aux_loss_weight > 0:
            for key, attr in auxiliary_shape_meta.items():
                dim = attr['shape'][0]
                self.aux_heads[key] = nn.Sequential(
                    nn.Linear(aux_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                )

    def encode_obs_features(
        self,
        nobs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        value = next(iter(nobs.values())) if isinstance(nobs, dict) else nobs
        B = value.shape[0]
        To = self.n_obs_steps

        if isinstance(nobs, dict):
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
        else:
            this_nobs = nobs[:, :To, ...].reshape(-1, *nobs.shape[2:])

        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def build_token_sequence(
        self,
        obs_features: torch.Tensor,
        action_prefix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        obs_tokens = self.obs_proj(obs_features)
        B = obs_tokens.shape[0]
        query_token = self.query_token.expand(B, -1, -1)
        tokens = [obs_tokens, query_token]
        if action_prefix is not None and action_prefix.shape[1] > 0:
            tokens.append(self.action_proj(action_prefix))
        return torch.cat(tokens, dim=1)

    def get_trunk_features(
        self,
        obs_features: torch.Tensor,
        action_prefix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        token_sequence = self.build_token_sequence(obs_features, action_prefix)
        return self.transformer(inputs_embeds=token_sequence).last_hidden_state

    def get_action_dist(self, h: torch.Tensor) -> Normal:
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(
            min=self.log_std_limits[0], max=self.log_std_limits[1]
        )
        return Normal(mean, torch.exp(log_std))

    def get_action_dist_from_prefix(
        self,
        obs_features: torch.Tensor,
        action_prefix: Optional[torch.Tensor] = None
    ) -> Normal:
        hidden = self.get_trunk_features(obs_features, action_prefix)
        prediction_hidden = hidden[:, self.n_obs_steps:, :]
        return self.get_action_dist(prediction_hidden)

    def forward(
        self,
        obs_features: torch.Tensor,
        action_prefix: Optional[torch.Tensor] = None
    ) -> Normal:
        return self.get_action_dist_from_prefix(obs_features, action_prefix)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        obs_features = self.encode_obs_features(nobs)

        predicted_actions = []
        action_prefix = None
        for _ in range(self.n_action_steps):
            dist = self.forward(obs_features, action_prefix)
            next_action = dist.mean[:, -1, :]
            predicted_actions.append(next_action)
            next_action_token = next_action.unsqueeze(1)
            if action_prefix is None:
                action_prefix = next_action_token
            else:
                action_prefix = torch.cat([action_prefix, next_action_token], dim=1)

        action_pred = torch.stack(predicted_actions, dim=1)
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
        obs_features = self.encode_obs_features(nobs)
        aux_input = obs_features.reshape(B, -1)

        start = To - 1
        end = start + Ta
        target = nactions[:, start:end]
        action_prefix = target[:, :-1] if Ta > 1 else None

        student_dist = self.forward(obs_features, action_prefix)

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
        aux_loss = torch.tensor(0.0, device=obs_features.device)
        if self.aux_heads and 'auxiliary_obs' in batch:
            for key, head in self.aux_heads.items():
                pred = head(aux_input)
                aux_target = self.normalizer[key].normalize(
                    batch['auxiliary_obs'][key][:, To-1])
                aux_loss = aux_loss + F.mse_loss(pred, aux_target)

        loss = bc_loss + self.aux_loss_weight * aux_loss
        return {'loss': loss, 'bc_loss': bc_loss, 'aux_loss': aux_loss}