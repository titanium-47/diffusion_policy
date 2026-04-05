from typing import Dict
import json
import hashlib
import shutil
from filelock import FileLock
from omegaconf import OmegaConf
import torch
import numpy as np
import copy
import zarr
import os
import gc
import shutil
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset

import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import glob


class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples
    copied from https://github.com/pytorch/pytorch/issues/2576
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        weights_np = self.weights.numpy()
        weights_sum = torch.sum(self.weights).numpy()
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=weights_np / weights_sum,
            replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def _compute_return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    returns = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns


class Sim2RealImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        use_cache: bool = False,
        use_disk: bool = True,
        add_returns: bool = False,
        gamma: float = 0.97,
        action_norm_mode: str = "limits",
    ):
        super().__init__()
        self.action_norm_mode = action_norm_mode
        self.dataset_path = dataset_path
        self.add_returns = add_returns
        self.gamma = float(gamma)
        assert os.path.isdir(dataset_path)
        print("shape_meta: ", shape_meta)
        # Load data and create replay buffer
        self.replay_buffer = self._create_replay_buffer_from_zarr(
            dataset_path, shape_meta=shape_meta, use_cache=use_cache,
            use_disk=use_disk)
        if self.add_returns:
            self._compute_returns()

        # replay_buffer = None
        # if use_cache:
        #     # fingerprint shape_meta
        #     shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
        #     shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
        #     cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
        #     cache_lock_path = cache_zarr_path + '.lock'
        #     print('Acquiring lock on cache.')
        #     with FileLock(cache_lock_path):
        #         if not os.path.exists(cache_zarr_path):
        #             # cache does not exists
        #             try:
        #                 print('Cache does not exist. Creating!')
        #                 replay_buffer = _get_replay_buffer( #doesn't exist in this file, so I've taken it from real_pusht_image_dataset.py
        #                     dataset_path=dataset_path,
        #                     shape_meta=shape_meta,
        #                     store=zarr.MemoryStore()
        #                 )
        #                 print('Saving cache to disk.')
        #                 with zarr.ZipStore(cache_zarr_path) as zip_store:
        #                     replay_buffer.save_to_store(
        #                         store=zip_store
        #                     )
        #             except Exception as e:
        #                 shutil.rmtree(cache_zarr_path)
        #                 raise e
        #         else:
        #             print('Loading cached ReplayBuffer from Disk.')
        #             with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
        #                 replay_buffer = ReplayBuffer.copy_from_store(
        #                     src_store=zip_store, store=zarr.MemoryStore())
        #             print('Loaded!')
        # else:
        #     replay_buffer = _get_replay_buffer(
        #         dataset_path=dataset_path,
        #         shape_meta=shape_meta,
        #         store=zarr.MemoryStore()
        #     )

        # Parse keys
        self.rgb_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'rgb']
        self.lowdim_keys = [
            k for k, v in shape_meta['obs'].items()
            if v.get('type', 'low_dim') == 'low_dim']
        self.auxiliary_keys = []
        if shape_meta.get('auxiliary_obs', None) is not None:
            self.auxiliary_keys = [
                k for k, v in shape_meta['auxiliary_obs'].items()]

        # Detect expert distribution keys (saved during data collection)
        self.expert_dist_keys = [
            k for k in self.replay_buffer.root['data'].keys()
            if k.startswith('expert_action_')]

        # Create key_first_k for performance optimization
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys + self.auxiliary_keys + self.expert_dist_keys:
                key_first_k[key] = n_obs_steps

        # Split train/val
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        # Store parameters
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_ratio = val_ratio
        self.seed = seed
        self.shape_meta = shape_meta
        self.val_mask = val_mask
        self.train_mask = train_mask

    def _compute_returns(self, replay_buffer=None):
        replay_buffer = self.replay_buffer if replay_buffer is None else replay_buffer
        if replay_buffer is None:
            raise ValueError("Replay buffer must be available to compute returns.")

        n_steps = int(replay_buffer.n_steps)
        data_root = replay_buffer.root['data']
        meta_root = replay_buffer.root['meta']
        cached_returns = data_root['returns_to_go'] if 'returns_to_go' in data_root else None
        cached_gamma = meta_root['return_to_go_gamma'] if 'return_to_go_gamma' in meta_root else None

        gamma_value = None
        if cached_gamma is not None:
            try:
                gamma_value = float(np.asarray(cached_gamma).reshape(()))
            except (TypeError, ValueError):
                gamma_value = None

        if cached_returns is not None:
            cached_shape = tuple(getattr(cached_returns, 'shape', ()))
            if cached_shape == (n_steps,) and gamma_value is not None and np.isclose(gamma_value, self.gamma):
                return

        if 'rewards' in replay_buffer:
            rewards = np.asarray(replay_buffer['rewards'], dtype=np.float32).reshape(-1)
        else:
            rewards = np.zeros((n_steps,), dtype=np.float32)

        returns = np.zeros((n_steps,), dtype=np.float32)
        start = 0
        for end in np.asarray(replay_buffer.episode_ends, dtype=np.int64):
            end = int(end)
            returns[start:end] = _compute_return_to_go(rewards[start:end], gamma=self.gamma)
            start = end

        if cached_returns is not None and isinstance(cached_returns, zarr.Array) and tuple(cached_returns.shape) == tuple(returns.shape):
            cached_returns[:] = returns
        else:
            if 'returns_to_go' in data_root and isinstance(data_root, zarr.Group):
                del data_root['returns_to_go']
            data_root['returns_to_go'] = returns

        gamma_array = np.array(self.gamma, dtype=np.float32)
        if cached_gamma is not None and isinstance(cached_gamma, zarr.Array) and tuple(cached_gamma.shape) == tuple(gamma_array.shape):
            cached_gamma[...] = gamma_array
        else:
            if 'return_to_go_gamma' in meta_root and isinstance(meta_root, zarr.Group):
                del meta_root['return_to_go_gamma']
            meta_root['return_to_go_gamma'] = gamma_array


    def _create_replay_buffer_from_zarr(
            self, dataset_path, shape_meta, use_cache=False, use_disk=True):
        """Create replay buffer from zarr data with caching and memory options."""
        replay_buffer = None

        if use_cache:
            # Create fingerprint for caching
            shape_meta_json = json.dumps(
                OmegaConf.to_container(shape_meta), sort_keys=True)
            dataset_stat = os.stat(dataset_path)
            cache_fingerprint = {
                'shape_meta': shape_meta_json,
                'dataset_path': dataset_path,
                'dataset_mtime': dataset_stat.st_mtime,
                'dataset_size': dataset_stat.st_size,
                'use_disk': use_disk,
                'add_returns': self.add_returns,
                'gamma': self.gamma if self.add_returns else None,
            }
            cache_fingerprint_json = json.dumps(
                cache_fingerprint, sort_keys=True)
            cache_hash = hashlib.md5(
                cache_fingerprint_json.encode('utf-8')).hexdigest()

            cache_zarr_path = os.path.join(
                dataset_path, cache_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'

            print(f'Acquiring lock on cache: {cache_zarr_path}')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # Cache does not exist, create it
                    try:
                        print('Cache does not exist. Creating!')
                        # Always load to memory for caching
                        replay_buffer = self._load_zarr_data(
                            dataset_path, use_disk=False)
                        if self.add_returns:
                            self._compute_returns(replay_buffer=replay_buffer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)

                        # Clear memory after cache creation
                        del replay_buffer
                        gc.collect()
                        replay_buffer = None
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from disk.')

                if replay_buffer is None:
                    # Load from cache
                    if use_disk:
                        # Load cache but keep it on disk (memory mapped)
                        zip_store = zarr.ZipStore(cache_zarr_path, mode='r')
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    else:
                        # Load cache into memory
                        with zarr.ZipStore(
                                cache_zarr_path, mode='r') as zip_store:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded cached data!')
        else:
            # No caching, load directly
            replay_buffer = self._load_zarr_data(
                dataset_path, use_disk=use_disk)
            if self.add_returns:
                self._compute_returns(replay_buffer=replay_buffer)

        return replay_buffer

    def _load_zarr_data(self, dataset_path, use_disk=True):
        """Load zarr data with disk/memory options"""
        z = zarr.open(dataset_path, mode='r')
        obs_group = z['data']['obs']
        action_arr = z['data']['actions']
        episode_ends = z['meta']['episode_ends']

        # Create replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Add observations
        for key in obs_group.keys():
            if use_disk:
                # Keep data on disk (memory mapped)
                replay_buffer.root['data'][key] = obs_group[key]
            else:
                # Load into memory
                replay_buffer.root['data'][key] = obs_group[key][:]

        # Always load to memory as it's small
        replay_buffer.root['data']['action'] = action_arr[:]
        if 'rewards' in z['data']:
            if use_disk:
                replay_buffer.root['data']['rewards'] = z['data']['rewards']
            else:
                replay_buffer.root['data']['rewards'] = z['data']['rewards'][:]
        replay_buffer.root['meta']['episode_ends'] = episode_ends[:]

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'], mode=self.action_norm_mode)
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key], mode="gaussian")
        # don't normalize rgb, obs_encoder has image_net norm
        for key in self.rgb_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        for key in self.auxiliary_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key], mode="gaussian")
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        n = self.n_obs_steps
        obs_slice = slice(0, n) if n is not None else slice(None)

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = (np.moveaxis(data[key][obs_slice], -1, 1)
                             .astype(np.float32) / 255.)
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][obs_slice].astype(np.float32)
            del data[key]

        aux_dict = dict()
        for key in self.auxiliary_keys:
            aux_dict[key] = data[key][obs_slice].astype(np.float32)
            del data[key]

        expert_dist = dict()
        for key in self.expert_dist_keys:
            expert_dist[key] = data[key][obs_slice].astype(np.float32)
            del data[key]

        action = data['action'].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
        }
        if aux_dict:
            torch_data['auxiliary_obs'] = dict_apply(aux_dict, torch.from_numpy)
        if expert_dist:
            torch_data['expert_dist'] = dict_apply(expert_dist, torch.from_numpy)
        if self.add_returns:
            torch_data['returns_to_go'] = torch.from_numpy(
                data['returns_to_go'].astype(np.float32))
        return torch_data


def discover_zarr_files(dataset_dir: str) -> list:
    """
    Discover all zarr files in a directory.
    Supports both .zarr directories and .zarr.zip files.
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    zarr_files = []

    # Find .zarr directories
    zarr_dirs = glob.glob(os.path.join(dataset_dir, "*.zarr"))
    zarr_files.extend([f for f in zarr_dirs if os.path.isdir(f)])

    # Find .zarr.zip files
    zarr_zips = glob.glob(os.path.join(dataset_dir, "*.zarr.zip"))
    zarr_files.extend(zarr_zips)

    if not zarr_files:
        raise ValueError(f"No zarr files found in directory: {dataset_dir}")

    # Sort for consistent ordering
    zarr_files.sort()

    print(f"Discovered {len(zarr_files)} zarr files in {dataset_dir}:")
    for f in zarr_files:
        print(f"  - {os.path.basename(f)}")

    return zarr_files


class FileStreamManager:
    """
    Manages file loading queue and double-buffering for streaming datasets.
    """
    def __init__(self, file_paths: list, 
                 samples_per_file_multiplier: float = 1.0):
        self.file_paths = file_paths
        self.samples_per_file_multiplier = samples_per_file_multiplier
        self.file_queue = deque(file_paths.copy())
        self.current_file_path = None
        self.next_file_path = None

        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.next_dataset_future = None
        self.lock = threading.Lock()

        # Shuffle initial queue
        import random
        file_list = list(self.file_queue)
        random.shuffle(file_list)
        self.file_queue = deque(file_list)

    def get_next_file_path(self) -> str:
        """Get next file path from queue, reshuffling if needed."""
        with self.lock:
            if not self.file_queue:
                # Reshuffle queue when empty
                file_list = self.file_paths.copy()
                import random
                random.shuffle(file_list)
                self.file_queue = deque(file_list)
                print("Reshuffled file queue for new epoch")

            return self.file_queue.popleft()

    def start_loading_next(self, dataset_kwargs: dict):
        """Start background loading of next dataset."""
        if self.next_dataset_future is not None:
            return  # Already loading

        self.next_file_path = self.get_next_file_path()
        dataset_kwargs = dataset_kwargs.copy()
        dataset_kwargs['dataset_path'] = self.next_file_path

        print(f"Starting background load of {self.next_file_path}")
        self.next_dataset_future = self.executor.submit(
            self._load_dataset, dataset_kwargs)

    def _load_dataset(self, dataset_kwargs: dict) -> 'Sim2RealImageDataset':
        """Load dataset in background thread."""
        return Sim2RealImageDataset(**dataset_kwargs)
    def get_next_dataset(self) -> 'Sim2RealImageDataset':
        """Get the next loaded dataset, blocking if necessary."""
        if self.next_dataset_future is None:
            raise RuntimeError("No dataset loading in progress")

        print(f"Waiting for {self.next_file_path} to finish loading...")
        dataset = self.next_dataset_future.result()
        print(f"Finished loading {self.next_file_path}")

        # Calculate samples for this file
        file_size = len(dataset)
        samples_for_file = int(file_size * self.samples_per_file_multiplier)

        # Reset for next load
        self.current_file_path = self.next_file_path
        self.next_file_path = None
        self.next_dataset_future = None

        return dataset, samples_for_file

    def shutdown(self):
        """Clean shutdown of background threads."""
        if self.next_dataset_future:
            self.next_dataset_future.cancel()
        self.executor.shutdown(wait=True)


class StreamingMultiDataset(BaseImageDataset):
    """
    Streaming multi-dataset with double-buffering and natural size-based
    sampling. Only keeps 2 datasets in memory at once, with background
    loading.
    """

    def __init__(
        self,
        file_paths: list = None,
        dataset_dir: str = None,
        shape_meta: dict = None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        use_cache: bool = False,
        use_disk: bool = False,
        samples_per_file_multiplier: float = 1.0,
        action_norm_mode: str = "limits",
        add_returns: bool = False,
        gamma: float = 0.97,
    ):
        super().__init__()

        # Handle file discovery
        if dataset_dir is not None:
            file_paths = discover_zarr_files(dataset_dir)
        elif file_paths is None:
            raise ValueError("Either file_paths or dataset_dir must be provided")

        # Store common dataset kwargs
        self.common_dataset_kwargs = {
            'shape_meta': shape_meta,
            'horizon': horizon,
            'pad_before': pad_before,
            'pad_after': pad_after,
            'n_obs_steps': n_obs_steps,
            'n_latency_steps': n_latency_steps,
            'seed': seed,
            'val_ratio': val_ratio,
            'use_cache': use_cache,
            'use_disk': use_disk,
            'action_norm_mode': action_norm_mode,
            'add_returns': add_returns,
            'gamma': gamma,
        }

        # Calculate total epoch length and normalizer from all files in one pass
        print("Loading all datasets to compute epoch length and normalizer...")
        self._compute_epoch_stats_and_normalizer(file_paths, **self.common_dataset_kwargs)

        # Initialize file manager
        self.file_manager = FileStreamManager(
            file_paths, samples_per_file_multiplier)

        # Double buffering state
        self.active_dataset = None
        self.standby_dataset = None
        self.samples_consumed_from_active = 0
        self.target_samples_for_active = 0

        # Initialize first two datasets
        self._initialize_datasets()

        # For compatibility with existing code
        self.shape_meta = shape_meta
        self.val_ratio = val_ratio

    def _compute_epoch_stats_and_normalizer(self, file_paths: list, **dataset_kwargs):
        """Compute total epoch length and normalizer from all datasets one at a time."""
        self.total_epoch_length = 0

        # Collect statistics from each dataset without keeping them all in memory
        dataset_stats = []
        dataset_sizes = []

        # Process datasets one at a time
        for file_path in file_paths:
            temp_kwargs = dataset_kwargs.copy()
            temp_kwargs['dataset_path'] = file_path
            temp_dataset = Sim2RealImageDataset(**temp_kwargs)

            # Accumulate epoch length
            file_length = len(temp_dataset)
            self.total_epoch_length += file_length
            dataset_sizes.append(file_length)

            # Get normalizer statistics
            dataset_normalizer = temp_dataset.get_normalizer()
            dataset_stats.append(dataset_normalizer)

            print(f"  {file_path}: {file_length} samples")

            # Clean up immediately to save memory
            del temp_dataset

        print(f"Total epoch length: {self.total_epoch_length} samples across {len(file_paths)} files")

        # Compute combined normalizer from collected statistics
        if len(dataset_stats) == 1:
            self._cached_normalizer = dataset_stats[0]
        else:
            # Combine normalizers using weighted averaging by dataset size
            combined_normalizer = LinearNormalizer()
            first_normalizer = dataset_stats[0]

            # Calculate weights based on dataset sizes
            total_size = sum(dataset_sizes)
            weights = [size / total_size for size in dataset_sizes]

            for key in first_normalizer.params_dict.keys():
                # Collect statistics from all normalizers for this key
                means = []
                stds = []

                for normalizer in dataset_stats:
                    field_normalizer = normalizer[key]
                    params = field_normalizer.params_dict
                    means.append(params['offset'])
                    stds.append(1.0 / params['scale'])

                # Convert to tensors for computation
                means = torch.stack(means)
                stds = torch.stack(stds)
                weights_tensor = torch.tensor(weights, dtype=torch.float32)

                # Compute weighted mean and std
                combined_mean = (means * weights_tensor.unsqueeze(-1)).sum(dim=0)
                combined_std = (stds * weights_tensor.unsqueeze(-1)).sum(dim=0)

                # Create combined normalizer for this key
                first_stats = dataset_stats[0][key].params_dict['input_stats']
                combined_normalizer[key] = (
                    SingleFieldLinearNormalizer.create_manual(
                        scale=1.0 / combined_std,
                        offset=combined_mean,
                        input_stats_dict=dict(first_stats)
                    ))

            self._cached_normalizer = combined_normalizer

        print("Epoch stats and normalizer computed from all datasets!")

    def _initialize_datasets(self):
        """Load first two datasets synchronously."""
        print("Initializing streaming datasets...")

        # Load first dataset
        first_path = self.file_manager.get_next_file_path()
        first_kwargs = self.common_dataset_kwargs.copy()
        first_kwargs['dataset_path'] = first_path
        self.active_dataset = Sim2RealImageDataset(**first_kwargs)
        self.target_samples_for_active = int(
            len(self.active_dataset) * 
            self.file_manager.samples_per_file_multiplier)
        print(f"Loaded active dataset: {first_path} "
              f"({len(self.active_dataset)} samples, "
              f"target: {self.target_samples_for_active})")

        # Start loading second dataset
        self.file_manager.start_loading_next(self.common_dataset_kwargs)

        # Load second dataset (will block)
        self.standby_dataset, standby_target = (
            self.file_manager.get_next_dataset())

        # Start loading third dataset in background
        self.file_manager.start_loading_next(self.common_dataset_kwargs)

        print("Streaming datasets initialized!")

    def _should_rotate_datasets(self) -> bool:
        """Check if we should rotate to next dataset."""
        return (self.samples_consumed_from_active >= 
                self.target_samples_for_active)

    def _rotate_datasets(self):
        """Rotate active/standby datasets and start loading next."""
        print(f"Rotating datasets (consumed {self.samples_consumed_from_active}"
              f"/{self.target_samples_for_active} from active)")

        # Swap active and standby
        old_active = self.active_dataset
        self.active_dataset = self.standby_dataset
        self.samples_consumed_from_active = 0
        self.target_samples_for_active = int(
            len(self.active_dataset) * 
            self.file_manager.samples_per_file_multiplier)

        # Get next dataset (blocks if not ready)
        self.standby_dataset, _ = self.file_manager.get_next_dataset()

        # Start loading the next one in background
        self.file_manager.start_loading_next(self.common_dataset_kwargs)

        # Clean up old dataset
        del old_active
        import gc
        gc.collect()

        print(f"Rotated to new active dataset "
              f"({len(self.active_dataset)} samples, "
              f"target: {self.target_samples_for_active})")

    def __len__(self):
        """Return total epoch length (sum of all datasets)."""
        return getattr(self, 'total_epoch_length', len(self.active_dataset) if self.active_dataset else 0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from active dataset and handle rotation."""
        if self._should_rotate_datasets():
            self._rotate_datasets()

        # Sample from active dataset
        # Use modulo to handle idx >= dataset length
        actual_idx = idx % len(self.active_dataset)
        sample = self.active_dataset[actual_idx]

        self.samples_consumed_from_active += 1
        return sample

    def get_validation_dataset(self):
        """Create validation version using active dataset."""
        # For validation, just use the active dataset's validation split
        return self.active_dataset.get_validation_dataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Get normalizer computed from all datasets during initialization."""
        return self._cached_normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Get actions from active dataset."""
        return self.active_dataset.get_all_actions()

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'file_manager'):
            self.file_manager.shutdown()


class Sim2RealImageMultiDataset(BaseImageDataset):
    """
    Multi-dataset wrapper that combines multiple Sim2RealImageDataset instances
    with weighted sampling.
    """

    def __init__(
        self,
        dataset_config: list = None,
        file_paths: list = None,
        dataset_dir: str = None,
        shape_meta: dict = None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        use_cache: bool = True,
        use_disk: bool = True,
        use_streaming: bool = False,
        samples_per_file_multiplier: float = 1.0,
        action_norm_mode: str = "limits",
        add_returns: bool = False,
        gamma: float = 0.97,
    ):
        super().__init__()

        # Handle different input formats
        if dataset_dir is not None:
            file_paths = discover_zarr_files(dataset_dir)
        elif dataset_config is not None:
            # Support both dataset_path and dataset_dir in config
            # Build expanded config with individual file paths
            expanded_config = []
            for config in dataset_config:
                sampling_ratio = config.get('sampling_ratio', 1.0)
                if 'dataset_path' in config:
                    expanded_config.append({
                        'dataset_path': config['dataset_path'],
                        'sampling_ratio': sampling_ratio
                    })
                elif 'dataset_dir' in config:
                    # Discover all zarr files in the directory
                    dir_files = discover_zarr_files(config['dataset_dir'])
                    # Split sampling ratio equally among files in directory
                    ratio_per_file = sampling_ratio / len(dir_files)
                    for file_path in dir_files:
                        expanded_config.append({
                            'dataset_path': file_path,
                            'sampling_ratio': ratio_per_file
                        })
                else:
                    raise ValueError(
                        "Each dataset_config entry must have either "
                        "'dataset_path' or 'dataset_dir'")

            # Extract file paths and update dataset_config
            file_paths = [config['dataset_path'] for config in expanded_config]
            dataset_config = expanded_config
        elif file_paths is None:
            raise ValueError(
                "Either dataset_dir, dataset_config, or file_paths must be "
                "provided")

        # Use streaming implementation if requested
        if use_streaming:
            self.streaming_dataset = StreamingMultiDataset(
                file_paths=file_paths,
                dataset_dir=None,  # Already resolved to file_paths above
                shape_meta=shape_meta,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                n_obs_steps=n_obs_steps,
                n_latency_steps=n_latency_steps,
                seed=seed,
                val_ratio=val_ratio,
                use_cache=use_cache,
                use_disk=use_disk,
                samples_per_file_multiplier=samples_per_file_multiplier,
                action_norm_mode=action_norm_mode,
                add_returns=add_returns,
                gamma=gamma,
            )
            self.is_streaming = True
            return

        # Fall back to original implementation
        self.is_streaming = False
        self.dataset_config = dataset_config or [
            {'dataset_path': path, 'sampling_ratio': 1.0} 
            for path in file_paths]
        self.shape_meta = shape_meta

        # Initialize datasets
        self.datasets = []
        self.dataset_lengths = []
        self.sampling_ratios = []

        common_kwargs = {
            'shape_meta': shape_meta,
            'horizon': horizon,
            'pad_before': pad_before,
            'pad_after': pad_after,
            'n_obs_steps': n_obs_steps,
            'n_latency_steps': n_latency_steps,
            'seed': seed,
            'val_ratio': val_ratio,
            'use_cache': use_cache,
            'use_disk': use_disk,
            'action_norm_mode': action_norm_mode,
            'add_returns': add_returns,
            'gamma': gamma,
        }

        for config in self.dataset_config:
            dataset_kwargs = {**common_kwargs}
            dataset_kwargs['dataset_path'] = config['dataset_path']

            dataset = Sim2RealImageDataset(**dataset_kwargs)

            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
            # Use natural sampling (size-based) if no ratio specified
            ratio = config.get('sampling_ratio', len(dataset))
            self.sampling_ratios.append(ratio)

        # Calculate cumulative indices for mapping global index to
        # (dataset_idx, local_idx)
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        self.total_length = int(self.cumulative_lengths[-1])

        # Calculate sampling weights
        self.weights = self._calculate_weights()

        # Create weighted sampler
        self.weighted_sampler = CustomWeightedRandomSampler(
            weights=self.weights, num_samples=self.total_length,
            replacement=True)

    def _calculate_weights(self) -> torch.Tensor:
        """Calculate per-sample weights based on dataset sizes and sampling ratios."""
        weights = []

        # Normalize sampling ratios
        total_ratio = sum(self.sampling_ratios)
        normalized_ratios = [r / total_ratio for r in self.sampling_ratios]

        for dataset_idx, (dataset_len, ratio) in enumerate(
                zip(self.dataset_lengths, normalized_ratios)):
            # Weight per sample = ratio / dataset_length
            sample_weight = ratio / dataset_len
            weights.extend([sample_weight] * dataset_len)

        return torch.tensor(weights, dtype=torch.float32)

    def _global_to_local_index(self, global_idx: int) -> tuple:
        """Convert global index to (dataset_idx, local_idx)."""
        dataset_idx = np.searchsorted(
            self.cumulative_lengths[1:], global_idx, side='right')
        local_idx = global_idx - self.cumulative_lengths[dataset_idx]
        return dataset_idx, local_idx

    def get_validation_dataset(self):
        """Create validation version of this multi-dataset wrapper."""
        if self.is_streaming:
            return self.streaming_dataset.get_validation_dataset()

        val_wrapper = copy.copy(self)
        val_wrapper.datasets = [
            dataset.get_validation_dataset() for dataset in self.datasets]

        # Recalculate lengths and weights for validation sets
        val_wrapper.dataset_lengths = [
            len(dataset) for dataset in val_wrapper.datasets]
        val_wrapper.cumulative_lengths = np.cumsum(
            [0] + val_wrapper.dataset_lengths)
        val_wrapper.total_length = int(val_wrapper.cumulative_lengths[-1])
        val_wrapper.weights = val_wrapper._calculate_weights()

        # Create new weighted sampler for validation
        val_wrapper.weighted_sampler = CustomWeightedRandomSampler(
            weights=val_wrapper.weights,
            num_samples=val_wrapper.total_length,
            replacement=True
        )

        return val_wrapper

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Combine normalizers from all datasets."""
        if self.is_streaming:
            return self.streaming_dataset.get_normalizer(**kwargs)

        # Get normalizers from all datasets
        normalizers = [
            dataset.get_normalizer(**kwargs) for dataset in self.datasets]

        if len(normalizers) == 1:
            return normalizers[0]

        # Combine normalizers using weighted averaging
        combined_normalizer = LinearNormalizer()

        # Get all keys from the first normalizer
        first_normalizer = normalizers[0]

        for key in first_normalizer.params_dict.keys():
            # Collect statistics from all normalizers for this key
            means = []
            stds = []
            weights = []

            for i, normalizer in enumerate(normalizers):
                field_normalizer = normalizer[key]
                # Access the parameters directly from params_dict
                params = field_normalizer.params_dict
                means.append(params['offset'])
                stds.append(1.0 / params['scale'])

                # Weight by sampling ratio (normalized) to match training distribution
                weights.append(self.sampling_ratios[i])

            # Convert to tensors for computation
            means = torch.stack(means)
            stds = torch.stack(stds)
            weights = torch.tensor(weights, dtype=torch.float32)

            # Normalize weights
            weights = weights / weights.sum()

            # Compute weighted mean and std
            combined_mean = (means * weights.unsqueeze(-1)).sum(dim=0)
            combined_std = (stds * weights.unsqueeze(-1)).sum(dim=0)

            # Create combined normalizer for this key
            # Use the first normalizer's input_stats as a template
            first_stats = normalizers[0][key].params_dict['input_stats']
            combined_normalizer[key] = (
                SingleFieldLinearNormalizer.create_manual(
                    scale=1.0 / combined_std,
                    offset=combined_mean,
                    input_stats_dict=dict(first_stats)
                ))

        return combined_normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Concatenate actions from all datasets."""
        if self.is_streaming:
            return self.streaming_dataset.get_all_actions()

        all_actions = []
        for dataset in self.datasets:
            all_actions.append(dataset.get_all_actions())
        return torch.cat(all_actions, dim=0)

    def __len__(self):
        if self.is_streaming:
            return len(self.streaming_dataset)
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_streaming:
            return self.streaming_dataset[idx]

        # For direct indexing, use the provided index
        dataset_idx, local_idx = self._global_to_local_index(idx)
        return self.datasets[dataset_idx][local_idx]

    def sample_weighted(self) -> Dict[str, torch.Tensor]:
        """Sample using the weighted sampler."""
        if self.is_streaming:
            # For streaming, just use regular indexing
            import random
            idx = random.randint(0, len(self) - 1)
            return self[idx]

        # Get a weighted sample index
        sampled_indices = list(self.weighted_sampler)
        global_idx = sampled_indices[0]  # Take first sample

        # Convert to local index and get sample
        dataset_idx, local_idx = self._global_to_local_index(global_idx)
        return self.datasets[dataset_idx][local_idx]
