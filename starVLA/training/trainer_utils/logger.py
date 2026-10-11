"""
Unified experiment tracking — supports SwanLab and WandB.

Usage in config YAML:
    tracker_type: swanlab       # "swanlab" (default) or "wandb"
    wandb_project: my_project   # used as project name for both backends
    wandb_entity: my_team       # used as workspace for SwanLab, entity for WandB

SwanLab 运行模式由环境变量 SWANLAB_MODE 控制 (默认 online):
    online     云端, 需先 `swanlab login`
    local      本地, 无需登录, 配合 `swanlab watch` 看面板
    disabled   关闭 tracker
train.sh 传 --local 等价于 SWANLAB_MODE=local。
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

from accelerate.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker(ABC):
    """Abstract experiment tracker."""

    @abstractmethod
    def init(self, cfg, output_dir, accelerator):
        ...

    @abstractmethod
    def log(self, metrics, step):
        ...

    @abstractmethod
    def finish(self):
        ...


class _SwanLabTracker(ExperimentTracker):
    def __init__(self):
        self._enabled = False
        self._swanlab = None

    def init(self, cfg, output_dir, accelerator):
        self._enabled = False
        # 运行模式: online(云端, 默认, 需 swanlab login) / local(本地, 无需登录) / disabled(关)
        # 由环境变量 SWANLAB_MODE 控制; train.sh 的 --local 即设 SWANLAB_MODE=local。
        mode = os.environ.get("SWANLAB_MODE", "online").lower()
        if mode == "disabled":
            accelerator.wait_for_everyone()
            return

        if accelerator.is_main_process:
            try:
                import swanlab

                self._swanlab = swanlab
                # online 模式需登录; swanlab 0.9.2 的 swanlab.login 不读环境变量,
                # 无 TTY (后台/nohup/slurm) 时交互登录失败 → init 退化成 "continuing without tracker"。
                # 若环境提供了 SWANLAB_API_KEY, 用它非交互 login + save (持久到 ~/.swanlab),
                # 之后所有 run 的 online init 都免登录、免 TTY。
                if mode == "online":
                    _key = os.environ.get("SWANLAB_API_KEY")
                    if _key:
                        try:
                            swanlab.login(api_key=_key, save=True)
                        except Exception as login_exc:
                            logger.warning(
                                f"swanlab.login(api_key from SWANLAB_API_KEY) failed: {login_exc}"
                            )
                swanlab.init(
                    project=cfg.wandb_project,
                    name=cfg.run_id,
                    description=getattr(cfg, "run_description", cfg.run_id),
                    log_dir=str(Path(output_dir) / "swanlog"),
                    mode=mode,
                )
                self._enabled = True
                logger.info(f"SwanLab initialized (mode={mode})")
            except ImportError:
                logger.warning(
                    "swanlab is not installed. Install with: pip install swanlab"
                )
            except Exception as exc:
                logger.warning(f"SwanLab init failed; continuing without tracker: {exc}")

        accelerator.wait_for_everyone()

    def log(self, metrics, step):
        if self._enabled and self._swanlab is not None:
            try:
                self._swanlab.log(metrics, step=step)
            except Exception as exc:
                self._enabled = False
                logger.warning(f"SwanLab log failed; disabling tracker: {exc}")

    def finish(self):
        if self._enabled and self._swanlab is not None:
            try:
                self._swanlab.finish()
            except Exception:
                pass


class _WandBTracker(ExperimentTracker):
    def __init__(self):
        self._enabled = False
        self._wandb = None

    def init(self, cfg, output_dir, accelerator):
        self._enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get(
            "WANDB_DISABLED", ""
        ).lower() in {"1", "true", "yes"}:
            accelerator.wait_for_everyone()
            return

        if accelerator.is_main_process:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    name=cfg.run_id,
                    dir=os.path.join(output_dir, "wandb"),
                    project=cfg.wandb_project,
                    entity=getattr(cfg, "wandb_entity", None),
                    group="vla-train",
                )
                self._enabled = True
                logger.info("W&B initialized successfully")
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")

        accelerator.wait_for_everyone()

    def log(self, metrics, step):
        if self._enabled and self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception as exc:
                self._enabled = False
                logger.warning(f"W&B log failed; disabling W&B: {exc}")

    def finish(self):
        if self._enabled and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


def build_tracker(cfg) -> ExperimentTracker:
    """Factory: return the tracker backend specified by ``cfg.tracker_type``.

    Defaults to ``wandb`` when the field is missing.
    """
    tracker_type = getattr(cfg, "tracker_type", "wandb")
    if tracker_type == "swanlab":
        return _SwanLabTracker()
    return _WandBTracker()
