import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _infer_hparams(cfg):
    am = getattr(getattr(cfg, "framework", None), "action_model", None)
    horizon = int(getattr(am, "action_horizon", 50)) if am else 50
    action_dim = int(getattr(am, "action_dim", 30)) if am else 30
    state_dim = int(getattr(am, "state_dim", 30)) if am else 30
    vd = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    bs = int(getattr(vd, "per_device_batch_size", 48)) if vd else 48
    try:
        n_views = 3
        if vd is not None:
            n_views = 3
    except Exception:
        n_views = 3
    return horizon, action_dim, state_dim, bs, n_views


class SyntheticH1Dataset(Dataset):
    def __init__(self, cfg=None, length=500000):
        self.cfg = cfg
        horizon, action_dim, state_dim, _, n_views = _infer_hparams(cfg)
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.length = int(length)
        base = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil = Image.fromarray(base)
        self._images = [pil.copy() for _ in range(n_views)]
        rng = np.random.default_rng(0)
        self._action = rng.uniform(-1, 1, size=(horizon, action_dim)).astype(np.float16)
        self._state = rng.uniform(-1, 1, size=(1, state_dim)).astype(np.float16)
        self._lang = "pick up the red block and place it on the table"
        self._provenance = {"dataset": "synthetic_h1", "episode": 0, "step": 0, "task": self._lang, "robot_tag": "NEW_EMBODIMENT"}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "action": self._action,
            "image": self._images,
            "lang": self._lang,
            "state": self._state,
            "robot_tag": "NEW_EMBODIMENT",
            "provenance": self._provenance,
        }

    def save_dataset_statistics(self, path):
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"synthetic_h1": {"note": "synthetic, no stats"}}, indent=2))


def collate_fn(batch):
    return batch


def make_synthetic_dataloader(cfg, mode="train"):
    vd = getattr(cfg.datasets, "vla_data", cfg.datasets.get("vla_data", {})) if hasattr(cfg, "datasets") else {}
    try:
        bs = int(vd.get("per_device_batch_size", 48)) if isinstance(vd, dict) else int(getattr(vd, "per_device_batch_size", 48))
    except Exception:
        bs = 48
    try:
        nw = int(vd.get("num_workers", 4)) if isinstance(vd, dict) else int(getattr(vd, "num_workers", 4))
    except Exception:
        nw = 4
    try:
        pf = int(vd.get("prefetch_factor", 2)) if isinstance(vd, dict) else int(getattr(vd, "prefetch_factor", 2))
    except Exception:
        pf = 2
    try:
        pm = bool(vd.get("pin_memory", True)) if isinstance(vd, dict) else bool(getattr(vd, "pin_memory", True))
    except Exception:
        pm = True
    try:
        pw = bool(vd.get("persistent_workers", True)) if isinstance(vd, dict) else bool(getattr(vd, "persistent_workers", True))
    except Exception:
        pw = True
    ds = SyntheticH1Dataset(cfg=cfg)
    kwargs = {"batch_size": bs, "collate_fn": collate_fn, "num_workers": nw, "pin_memory": pm}
    if nw > 0:
        kwargs["persistent_workers"] = pw
        kwargs["prefetch_factor"] = pf
    return DataLoader(ds, **kwargs)
