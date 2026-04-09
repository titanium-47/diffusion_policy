from __future__ import annotations

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import json
import os
import pathlib
import random
from typing import Any, Dict, cast

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.policy.q_image_policy import QImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainBellmanQWorkspace(BaseWorkspace):
    include_keys = [
        "global_step",
        "epoch",
        "best_val_loss",
        "history",
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        cfg_any = cast(Any, cfg)

        seed = cfg_any.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: QImagePolicy = hydra.utils.instantiate(cfg_any.policy)
        self.target_model: QImagePolicy = hydra.utils.instantiate(cfg_any.policy)
        self._hard_update_target()
        self.target_model.requires_grad_(False)

        self.q_optimizer = hydra.utils.instantiate(
            cfg_any.q_optimizer,
            params=self._trainable_q_parameters(),
        )

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.history = {
            "q_train_loss": [],
            "q_val_loss": [],
        }

    def _trainable_q_parameters(self):
        return list(self.model.obs_encoder.parameters()) + list(self.model.q_head.parameters())

    def _move_batch_to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return dict_apply(batch, lambda x: x.to(device, non_blocking=True))

    def _sanitize_dataloader_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = dict(kwargs)
        if int(kwargs.get("num_workers", 0)) == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
        return kwargs

    def _best_checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "checkpoints", "q_best.ckpt")

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"q_epoch_{epoch:04d}.ckpt")

    def _hard_update_target(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target(self, tau: float) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def _set_train_mode(self) -> None:
        self.model.train()
        self.model.obs_encoder.train()
        self.model.q_head.train()
        self.model.value_head.eval()
        self.model.obs_encoder.requires_grad_(True)
        self.model.q_head.requires_grad_(True)
        self.model.value_head.requires_grad_(False)

    def _set_eval_mode(self) -> None:
        self.model.eval()
        self.model.obs_encoder.requires_grad_(True)
        self.model.q_head.requires_grad_(True)
        self.model.value_head.requires_grad_(False)
        self.target_model.eval()

    def _metrics_to_float(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                result[key] = float(value.detach().item())
            elif isinstance(value, (float, int)):
                result[key] = float(value)
        return result

    def _prefix_metrics(self, split: str, metrics: Dict[str, float]) -> Dict[str, float]:
        return {f"q/{split}_{key}": value for key, value in metrics.items()}

    def _maybe_save_checkpoint(self, epoch: int) -> None:
        cfg = cast(Any, self.cfg)
        checkpoint_every = int(cfg.training.checkpoint_every)
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            self.save_checkpoint(path=self._epoch_checkpoint_path(epoch), use_thread=False)
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint(use_thread=False)

    def _write_metrics(self) -> None:
        metrics_path = os.path.join(self.output_dir, "bellman_q_training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **self.history,
                    "best_val_loss": self.best_val_loss,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                },
                f,
                indent=2,
            )

    def _slice_obs_window(
        self,
        obs_dict: Dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> Dict[str, torch.Tensor]:
        return {key: value[:, start:end, ...] for key, value in obs_dict.items()}

    def _get_batch_lengths(self, batch: Dict[str, Any]) -> tuple[int, int, int]:
        obs_length = min(value.shape[1] for value in batch["obs"].values())
        action_length = int(batch["action"].shape[1])
        reward_length = int(batch["rewards"].shape[1])
        return obs_length, action_length, reward_length

    def _extract_bellman_batch(self, batch: Dict[str, Any]):
        cfg = cast(Any, self.cfg)
        k = int(cfg.n_obs_steps)
        n = int(cfg.n_action_steps)
        obs_length, action_length, reward_length = self._get_batch_lengths(batch)

        required_obs = k + n
        required_action = k + 2 * n - 1
        required_reward = k + n - 1
        if obs_length < required_obs:
            raise ValueError(
                f"Bellman Q requires obs length >= {required_obs}, got {obs_length}. "
                f"Set dataset_obs_steps high enough (typically to horizon={cfg.horizon})."
            )
        if action_length < required_action:
            raise ValueError(
                f"Bellman Q requires action length >= {required_action}, got {action_length}. "
                f"Increase horizon to at least n_obs_steps + 2 * n_action_steps - 1."
            )
        if reward_length < required_reward:
            raise ValueError(
                f"Bellman Q requires reward length >= {required_reward}, got {reward_length}. "
                f"Ensure the dataset emits rewards across the configured horizon."
            )

        obs = batch["obs"]
        action = batch["action"]
        rewards = batch["rewards"]

        current_obs = self._slice_obs_window(obs, 0, k)
        current_action = action[:, (k - 1) : (k + n - 1), ...]
        next_obs = self._slice_obs_window(obs, n, k + n)
        next_action = action[:, (k + n - 1) : (k + 2 * n - 1), ...]
        reward_segment = rewards[:, (k - 1) : (k + n - 1)]
        return current_obs, current_action, next_obs, next_action, reward_segment

    def _compute_discounted_reward_sum(self, reward_segment: torch.Tensor) -> torch.Tensor:
        cfg = cast(Any, self.cfg)
        discounts = torch.pow(
            torch.as_tensor(float(cfg.gamma), device=reward_segment.device, dtype=reward_segment.dtype),
            torch.arange(reward_segment.shape[1], device=reward_segment.device, dtype=reward_segment.dtype),
        )
        reward_sum = (reward_segment * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)
        return reward_sum

    def _compute_bellman_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cfg = cast(Any, self.cfg)
        current_obs, current_action, next_obs, next_action, reward_segment = self._extract_bellman_batch(batch)

        pred_q = self.model.predict_q(current_obs, current_action)
        reward_sum = self._compute_discounted_reward_sum(reward_segment)

        with torch.no_grad():
            target_bootstrap = self.target_model.predict_q(next_obs, next_action)
            target_q = reward_sum + (float(cfg.gamma) ** int(cfg.n_action_steps)) * target_bootstrap

        loss = F.mse_loss(pred_q, target_q)
        return {
            "loss": loss,
            "q_loss": loss,
            "q_pred_mean": pred_q.mean(),
            "target_q_mean": target_q.mean(),
            "reward_sum_mean": reward_sum.mean(),
            "target_bootstrap_mean": target_bootstrap.mean(),
        }

    def _run_epoch(
        self,
        dataloader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        max_train_steps: int | None = None,
    ) -> Dict[str, float]:
        self._set_train_mode()
        self.target_model.eval()
        metric_totals: Dict[str, float] = {}
        num_batches = 0
        self.model.to(device)
        self.target_model.to(device)
        cfg = cast(Any, self.cfg)
        target_tau = float(cfg.training.target_tau)
        target_update_every = max(int(cfg.training.target_update_every), 1)

        with tqdm.tqdm(
            dataloader,
            desc="Bellman Q epoch",
            leave=False,
            mininterval=cfg.training.tqdm_interval_sec,
        ) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = self._move_batch_to_device(batch, device)
                loss_output = self._compute_bellman_loss(batch)
                metrics = self._metrics_to_float(loss_output)
                loss = loss_output["loss"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_norm = float(cfg.training.gradient_clip_norm)
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._trainable_q_parameters(), max_norm=clip_norm)
                optimizer.step()

                if (self.global_step + 1) % target_update_every == 0:
                    self._soft_update_target(target_tau)

                for key, value in metrics.items():
                    metric_totals[key] = metric_totals.get(key, 0.0) + value
                num_batches += 1
                tepoch.set_postfix(loss=metrics["loss"], refresh=False)
                self.global_step += 1

                if max_train_steps is not None and batch_idx >= (max_train_steps - 1):
                    break

        if num_batches == 0:
            return {"loss": 0.0}
        return {key: value / num_batches for key, value in metric_totals.items()}

    def _evaluate(
        self,
        dataloader: DataLoader,
        device: torch.device,
        max_val_steps: int | None = None,
    ) -> Dict[str, float]:
        self._set_eval_mode()
        metric_totals: Dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            with tqdm.tqdm(
                dataloader,
                desc="Bellman Q validation",
                leave=False,
                mininterval=cast(Any, self.cfg).training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = self._move_batch_to_device(batch, device)
                    loss_output = self._compute_bellman_loss(batch)
                    metrics = self._metrics_to_float(loss_output)
                    for key, value in metrics.items():
                        metric_totals[key] = metric_totals.get(key, 0.0) + value
                    num_batches += 1
                    if max_val_steps is not None and batch_idx >= (max_val_steps - 1):
                        break

        if num_batches == 0:
            return {"loss": 0.0}
        return {key: value / num_batches for key, value in metric_totals.items()}

    def run(self):
        cfg = cast(Any, copy.deepcopy(self.cfg))
        os.makedirs(self.output_dir, exist_ok=True)
        if not bool(cfg.add_returns):
            raise ValueError("Bellman Q training requires add_returns=True.")

        checkpoint_loaded = False
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                self.load_checkpoint(path=latest_ckpt_path)
                checkpoint_loaded = True

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("Instantiated image dataset does not implement BaseImageDataset.")

        dataloader_kwargs = self._sanitize_dataloader_kwargs(dict(cfg.dataloader))
        weighted_sampler = getattr(dataset, "weighted_sampler", None)
        if weighted_sampler is not None:
            dataloader_kwargs["sampler"] = weighted_sampler
            dataloader_kwargs.pop("shuffle", None)

        train_dataloader = DataLoader(dataset, **dataloader_kwargs)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            **self._sanitize_dataloader_kwargs(dict(cfg.val_dataloader)),
        )

        if checkpoint_loaded and len(self.model.normalizer.params_dict) > 0:
            normalizer = self.model.normalizer
        else:
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            self.target_model.set_normalizer(normalizer)
            self._hard_update_target()

        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.target_model.to(device)
        optimizer_to(self.q_optimizer, device)

        if cfg.training.debug:
            cfg.training.epochs = min(int(cfg.training.epochs), 2)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=cast(Any, OmegaConf.to_container(cfg, resolve=True)),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir}, allow_val_change=True)

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        max_train_steps = cfg.training.max_train_steps
        max_val_steps = cfg.training.max_val_steps

        with JsonLogger(log_path) as json_logger:
            for epoch in range(self.epoch + 1, int(cfg.training.epochs) + 1):
                train_loss = self._run_epoch(
                    train_dataloader,
                    device,
                    self.q_optimizer,
                    max_train_steps=max_train_steps,
                )
                val_loss = self._evaluate(
                    val_dataloader,
                    device,
                    max_val_steps=max_val_steps,
                )

                self.epoch = epoch
                self.history["q_train_loss"].append(train_loss["loss"])
                self.history["q_val_loss"].append(val_loss["loss"])

                if val_loss["loss"] < self.best_val_loss:
                    self.best_val_loss = val_loss["loss"]
                    self.save_checkpoint(
                        path=self._best_checkpoint_path(),
                        use_thread=False,
                    )

                step_log = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    **self._prefix_metrics("train", train_loss),
                    **self._prefix_metrics("val", val_loss),
                    "q/best_val_loss": self.best_val_loss,
                }
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self._write_metrics()
                self._maybe_save_checkpoint(epoch)

        wandb_run.summary["best_val_loss"] = self.best_val_loss
        wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_bellman_q_sim2real_state_workspace",
)
def main(cfg):
    workspace = TrainBellmanQWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
