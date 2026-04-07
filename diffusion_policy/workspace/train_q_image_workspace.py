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

import dill
import hydra
import numpy as np
import torch
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


class TrainQImageWorkspace(BaseWorkspace):
    include_keys = [
        "global_step",
        "stage",
        "value_epoch",
        "q_epoch",
        "best_value_val_loss",
        "best_q_val_loss",
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
        self.value_optimizer = hydra.utils.instantiate(
            cfg_any.value_optimizer,
            params=self.model.value_parameters(),
        )
        self.q_optimizer = hydra.utils.instantiate(
            cfg_any.q_optimizer,
            params=self.model.q_parameters(),
        )

        self.global_step = 0
        self.stage = "value"
        self.value_epoch = 0
        self.q_epoch = 0
        self.best_value_val_loss = float("inf")
        self.best_q_val_loss = float("inf")
        self.history = {
            "value_train_loss": [],
            "value_val_loss": [],
            "q_train_loss": [],
            "q_val_loss": [],
        }

    def _move_batch_to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return dict_apply(batch, lambda x: x.to(device, non_blocking=True))

    def _sanitize_dataloader_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = dict(kwargs)
        if int(kwargs.get("num_workers", 0)) == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
        return kwargs

    def _best_checkpoint_path(self, stage: str) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"{stage}_best.ckpt")

    def _stage_checkpoint_path(self, stage: str, epoch: int) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"{stage}_epoch_{epoch:04d}.ckpt")

    def _load_pretrained_value_weights(self, ckpt_path: str) -> None:
        """Load obs_encoder, value_head, and normalizer from another Q-training checkpoint.

        Used to warm-start (or freeze via epochs/hyperparams) the value function before Q training.
        Expects a standard workspace .ckpt with ``state_dicts['model']``.
        """
        path = pathlib.Path(ckpt_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"preload_value_checkpoint does not exist: {path}")

        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        if "state_dicts" not in payload or "model" not in payload["state_dicts"]:
            raise KeyError(f"Checkpoint at {path} has no state_dicts['model'] (not a workspace ckpt?)")

        src_sd = payload["state_dicts"]["model"]
        prefixes = ("obs_encoder.", "value_head.", "normalizer.")
        filtered = {k: v for k, v in src_sd.items() if k.startswith(prefixes)}
        if not filtered:
            raise RuntimeError(f"No keys matching {prefixes} in {path}")

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading value weights from {path}: {unexpected[:8]}")

        value_missing = [k for k in missing if k.startswith(prefixes)]
        if value_missing:
            print(
                f"[TrainQImageWorkspace] Warning: checkpoint missing {len(value_missing)} value-related "
                f"params (shape/arch mismatch?). Examples: {value_missing[:5]}"
            )

    def _metrics_to_float(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                result[key] = float(value.detach().item())
            elif isinstance(value, (float, int)):
                result[key] = float(value)
        return result

    def _prefix_metrics(self, stage: str, split: str, metrics: Dict[str, float]) -> Dict[str, float]:
        return {f"{stage}/{split}_{key}": value for key, value in metrics.items()}

    def _maybe_save_checkpoint(self, stage: str, epoch: int):
        cfg = cast(Any, self.cfg)
        checkpoint_every = int(cfg.training.checkpoint_every)
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            self.save_checkpoint(path=self._stage_checkpoint_path(stage, epoch), use_thread=False)
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint(use_thread=False)

    def _run_epoch(
        self,
        dataloader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        stage: str,
        max_train_steps: int | None = None,
    ) -> Dict[str, float]:
        if stage == "value":
            self.model.set_value_training_mode()
            compute_loss = self.model.compute_value_loss
        else:
            self.model.set_q_training_mode()
            compute_loss = self.model.compute_q_loss

        metric_totals: Dict[str, float] = {}
        num_batches = 0
        self.model.to(device)

        with tqdm.tqdm(
            dataloader,
            desc=f"{stage.capitalize()} epoch",
            leave=False,
            mininterval=cast(Any, self.cfg).training.tqdm_interval_sec,
        ) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = self._move_batch_to_device(batch, device)
                loss_output = compute_loss(batch)
                metrics = self._metrics_to_float(loss_output)
                loss = loss_output["loss"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_norm = float(cast(Any, self.cfg).training.gradient_clip_norm)
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=clip_norm,
                    )
                optimizer.step()

                loss_cpu = metrics["loss"]
                for key, value in metrics.items():
                    metric_totals[key] = metric_totals.get(key, 0.0) + value
                num_batches += 1
                tepoch.set_postfix(loss=loss_cpu, refresh=False)
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
        stage: str,
        max_val_steps: int | None = None,
    ) -> Dict[str, float]:
        if stage == "value":
            self.model.set_value_training_mode()
            compute_loss = self.model.compute_value_loss
        else:
            self.model.set_q_training_mode()
            compute_loss = self.model.compute_q_loss

        metric_totals: Dict[str, float] = {}
        num_batches = 0
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(
                dataloader,
                desc=f"{stage.capitalize()} validation",
                leave=False,
                mininterval=cast(Any, self.cfg).training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = self._move_batch_to_device(batch, device)
                    loss_output = compute_loss(batch)
                    metrics = self._metrics_to_float(loss_output)
                    for key, value in metrics.items():
                        metric_totals[key] = metric_totals.get(key, 0.0) + value
                    num_batches += 1
                    if max_val_steps is not None and batch_idx >= (max_val_steps - 1):
                        break

        if num_batches == 0:
            return {"loss": 0.0}
        return {key: value / num_batches for key, value in metric_totals.items()}

    def _write_metrics(self):
        metrics_path = os.path.join(self.output_dir, "q_training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **self.history,
                    "best_value_val_loss": self.best_value_val_loss,
                    "best_q_val_loss": self.best_q_val_loss,
                    "global_step": self.global_step,
                    "stage": self.stage,
                    "value_epoch": self.value_epoch,
                    "q_epoch": self.q_epoch,
                },
                f,
                indent=2,
            )

    def run(self):
        cfg = cast(Any, copy.deepcopy(self.cfg))
        os.makedirs(self.output_dir, exist_ok=True)
        if not bool(cfg.add_returns):
            raise ValueError("Q-image training requires add_returns=True.")

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

        preload_path = getattr(cfg.training, "preload_value_checkpoint", None)
        if preload_path and not checkpoint_loaded:
            self._load_pretrained_value_weights(str(preload_path))
        elif preload_path and checkpoint_loaded:
            print(
                "[TrainQImageWorkspace] Skipping preload_value_checkpoint because training.resume "
                "loaded a checkpoint (model state already restored)."
            )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.value_optimizer, device)
        optimizer_to(self.q_optimizer, device)

        if cfg.training.debug:
            cfg.training.value_epochs = min(int(cfg.training.value_epochs), 2)
            cfg.training.q_epochs = min(int(cfg.training.q_epochs), 2)
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
            if self.stage == "value":
                for epoch in range(self.value_epoch + 1, int(cfg.training.value_epochs) + 1):
                    train_loss = self._run_epoch(
                        train_dataloader,
                        device,
                        self.value_optimizer,
                        stage="value",
                        max_train_steps=max_train_steps,
                    )
                    val_loss = self._evaluate(
                        val_dataloader,
                        device,
                        stage="value",
                        max_val_steps=max_val_steps,
                    )

                    self.value_epoch = epoch
                    self.history["value_train_loss"].append(train_loss["loss"])
                    self.history["value_val_loss"].append(val_loss["loss"])

                    if val_loss["loss"] < self.best_value_val_loss:
                        self.best_value_val_loss = val_loss["loss"]
                        self.save_checkpoint(
                            path=self._best_checkpoint_path("value"),
                            use_thread=False,
                        )

                    step_log = {
                        "stage": "value",
                        "epoch": epoch,
                        "global_step": self.global_step,
                        **self._prefix_metrics("value", "train", train_loss),
                        **self._prefix_metrics("value", "val", val_loss),
                        "value/best_val_loss": self.best_value_val_loss,
                    }
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self._write_metrics()
                    self._maybe_save_checkpoint("value", epoch)

                best_value_path = self._best_checkpoint_path("value")
                if os.path.exists(best_value_path):
                    self.load_checkpoint(path=best_value_path)
                self.stage = "q"
                self.q_epoch = 0

            if self.stage == "q":
                for epoch in range(self.q_epoch + 1, int(cfg.training.q_epochs) + 1):
                    train_loss = self._run_epoch(
                        train_dataloader,
                        device,
                        self.q_optimizer,
                        stage="q",
                        max_train_steps=max_train_steps,
                    )
                    val_loss = self._evaluate(
                        val_dataloader,
                        device,
                        stage="q",
                        max_val_steps=max_val_steps,
                    )

                    self.q_epoch = epoch
                    self.history["q_train_loss"].append(train_loss["loss"])
                    self.history["q_val_loss"].append(val_loss["loss"])

                    if val_loss["loss"] < self.best_q_val_loss:
                        self.best_q_val_loss = val_loss["loss"]
                        self.save_checkpoint(
                            path=self._best_checkpoint_path("q"),
                            use_thread=False,
                        )

                    step_log = {
                        "stage": "q",
                        "epoch": epoch,
                        "global_step": self.global_step,
                        **self._prefix_metrics("q", "train", train_loss),
                        **self._prefix_metrics("q", "val", val_loss),
                        "q/best_val_loss": self.best_q_val_loss,
                    }
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self._write_metrics()
                    self._maybe_save_checkpoint("q", epoch)

        wandb_run.summary["best_value_val_loss"] = self.best_value_val_loss
        wandb_run.summary["best_q_val_loss"] = self.best_q_val_loss
        wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_q_sim2real_state_workspace",
)
def main(cfg):
    workspace = TrainQImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
