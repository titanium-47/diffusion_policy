"""Dataset for loading UWLab .pt trajectory files (low-dim / state only)."""

from typing import Dict, List, Optional
import copy
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)


def _find_obs_value(traj_data: dict, key: str):
    """Search for an observation key across all UWLab trajectory obs dicts."""
    for group in ("obs_proprio", "obs_assets", "obs_other_state"):
        container = traj_data.get(group, {})
        if isinstance(container, dict) and key in container:
            v = container[key]
            return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    return None


class UwlabTrajectoryLowdimDataset(BaseLowdimDataset):
    """Loads UWLab `.pt` trajectory data for state-conditioned diffusion policy training.

    State observations are extracted from ``obs_proprio``, ``obs_assets``, and
    ``obs_other_state`` by the keys listed in ``obs_keys``, then concatenated
    into a flat obs vector.
    """

    DEFAULT_OBS_KEYS: List[str] = [
        "prev_actions",
        "joint_pos",
        "end_effector_pose",
        "insertive_asset_pose",
        "receptive_asset_pose",
        "insertive_asset_in_receptive_asset_frame",
    ]

    def __init__(
        self,
        dataset_dir: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        obs_keys: Optional[List[str]] = None,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
    ):
        dataset_dir = Path(dataset_dir)
        assert dataset_dir.is_dir(), f"Dataset directory not found: {dataset_dir}"

        if obs_keys is None:
            obs_keys = list(self.DEFAULT_OBS_KEYS)
        obs_keys = list(obs_keys)

        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        traj_files = manifest.get("files", [])
        for traj_info in tqdm(traj_files, desc="Loading UWLab lowdim trajectories"):
            traj_path = dataset_dir / traj_info["file"]
            traj_data = torch.load(traj_path, map_location="cpu")

            actions = traj_data["actions"]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()

            obs_parts = []
            for key in obs_keys:
                value = _find_obs_value(traj_data, key)
                if value is None:
                    raise KeyError(
                        f"Obs key '{key}' not found in {traj_path}. "
                        f"Available proprio={list(traj_data.get('obs_proprio', {}).keys())}, "
                        f"assets={list(traj_data.get('obs_assets', {}).keys())}, "
                        f"other={list(traj_data.get('obs_other_state', {}).keys())}"
                    )
                obs_parts.append(value.astype(np.float32))

            obs = np.concatenate(obs_parts, axis=-1)
            episode = {
                "obs": obs,
                "action": actions.astype(np.float32),
            }
            replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.obs_keys = obs_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        normalizer["obs"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["obs"]
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
