"""Physics-guided residual neural process model for Phase C."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class PGRNPConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    seed: int = 42
    hidden_dim: int = 96
    latent_dim: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.10
    max_context: int = 64
    likelihood: str = "student_t"
    lambda_kl: float = 0.02
    lambda_bound: float = 0.02
    lambda_smooth: float = 0.005
    train_ratio: float = 0.70
    calibration_ratio: float = 0.15


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_bs(df: pd.DataFrame, config: PGRNPConfig) -> Dict[str, List[str]]:
    rng = np.random.default_rng(config.seed)
    bs = np.array(sorted(df["BS"].astype(str).unique()))
    rng.shuffle(bs)
    n_train = int(len(bs) * config.train_ratio)
    n_cal = int(len(bs) * config.calibration_ratio)
    return {"train": bs[:n_train].tolist(), "calibration": bs[n_train:n_train + n_cal].tolist(), "test": bs[n_train + n_cal:].tolist()}


class ResidualTrajectoryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        bs_list: Iterable[str],
        feature_scaler: StandardScaler,
        residual_mean: float,
        residual_std: float,
        config: PGRNPConfig,
        train_mode: bool = True,
    ):
        self.df = df[df["BS"].astype(str).isin(set(map(str, bs_list)))].copy()
        # Keep the caller's row identity across split-specific reset_index().
        # Predictions are merged back to the full residual dataset by this id.
        if "_row_id" not in self.df.columns:
            self.df["_row_id"] = self.df.index.astype(np.int64)
        self.feature_cols = feature_cols
        self.feature_scaler = feature_scaler
        self.residual_mean = residual_mean
        self.residual_std = max(float(residual_std), 1e-6)
        self.config = config
        self.train_mode = train_mode
        if self.df.empty:
            # Allow empty splits (e.g., tiny smoke tests) without crashing sklearn's scaler.
            self.X = np.empty((0, len(self.feature_cols)), dtype=np.float32)
            self.y_scaled = np.empty((0,), dtype=np.float32)
            self.groups = []
            self.bs_to_rows = {}
            return

        self.df["origin_time"] = pd.to_datetime(self.df["origin_time"])
        self.df["target_time"] = pd.to_datetime(self.df["target_time"])
        self.df = self.df.sort_values(["BS", "origin_time", "horizon"]).reset_index(drop=True)
        x_raw = self.df[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        self.X = self.feature_scaler.transform(x_raw).astype(np.float32)
        self.y_scaled = (
            (self.df["residual_true"].to_numpy(dtype=np.float32) - self.residual_mean) / self.residual_std
        ).astype(np.float32)
        self.groups = list(self.df.groupby(["BS", "origin_time"], sort=False).indices.items())
        self.bs_to_rows = {str(bs): g.index.to_numpy(dtype=np.int64) for bs, g in self.df.groupby("BS", sort=False)}

    def __len__(self) -> int:
        return len(self.groups)

    def _context_indices(self, bs: str, origin_time: pd.Timestamp, target_rows: np.ndarray) -> np.ndarray:
        rows = self.bs_to_rows.get(str(bs), np.array([], dtype=np.int64))
        if len(rows) == 0:
            return np.array([], dtype=np.int64)
        hist_mask = self.df.loc[rows, "target_time"].to_numpy() < np.datetime64(origin_time)
        ctx = rows[hist_mask]
        if len(ctx) == 0:
            ctx = np.setdiff1d(rows, target_rows, assume_unique=False)
        if len(ctx) == 0:
            return np.array([], dtype=np.int64)
        max_ctx = min(self.config.max_context, len(ctx))
        if self.train_mode and len(ctx) > max_ctx:
            return np.random.choice(ctx, size=max_ctx, replace=False).astype(np.int64)
        return np.asarray(ctx[-max_ctx:], dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        (bs, origin_time), row_idx = self.groups[idx]
        row_idx = np.asarray(row_idx, dtype=np.int64)
        ctx_idx = self._context_indices(str(bs), pd.Timestamp(origin_time), row_idx)
        d = len(self.feature_cols)
        target_x = np.zeros((24, d), dtype=np.float32)
        target_y = np.zeros(24, dtype=np.float32)
        mask = np.zeros(24, dtype=np.float32)
        e_phys = np.zeros(24, dtype=np.float32)
        lower = np.zeros(24, dtype=np.float32)
        upper = np.zeros(24, dtype=np.float32)
        row_ids = np.full(24, -1, dtype=np.int64)

        for rid in row_idx:
            pos = int(self.df.loc[rid, "horizon"]) - 1
            target_x[pos] = self.X[rid]
            target_y[pos] = self.y_scaled[rid]
            mask[pos] = 1.0
            e_phys[pos] = float(self.df.loc[rid, "E_phys_hat"])
            lower[pos] = float(self.df.loc[rid, "physical_lower"])
            upper[pos] = float(self.df.loc[rid, "physical_upper"])
            row_ids[pos] = int(self.df.loc[rid, "_row_id"])

        if len(ctx_idx) == 0:
            ctx_x = np.zeros((1, d), dtype=np.float32)
            ctx_y = np.zeros(1, dtype=np.float32)
            ctx_mask = np.zeros(1, dtype=np.float32)
        else:
            ctx_x = self.X[ctx_idx]
            ctx_y = self.y_scaled[ctx_idx]
            ctx_mask = np.ones(len(ctx_idx), dtype=np.float32)

        return {
            "context_x": torch.tensor(ctx_x), "context_y": torch.tensor(ctx_y), "context_mask": torch.tensor(ctx_mask),
            "target_x": torch.tensor(target_x), "target_y": torch.tensor(target_y), "target_mask": torch.tensor(mask),
            "e_phys": torch.tensor(e_phys), "physical_lower": torch.tensor(lower), "physical_upper": torch.tensor(upper),
            "row_ids": torch.tensor(row_ids, dtype=torch.long), "BS": str(bs), "origin_time": str(origin_time),
        }


def collate_trajectories(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    max_ctx = max(int(x["context_x"].shape[0]) for x in batch)  # type: ignore[index]
    d = int(batch[0]["target_x"].shape[-1])  # type: ignore[index]
    bsz = len(batch)
    out: Dict[str, torch.Tensor | List[str]] = {
        "context_x": torch.zeros(bsz, max_ctx, d),
        "context_y": torch.zeros(bsz, max_ctx),
        "context_mask": torch.zeros(bsz, max_ctx),
        "target_x": torch.stack([x["target_x"] for x in batch]),  # type: ignore[list-item]
        "target_y": torch.stack([x["target_y"] for x in batch]),  # type: ignore[list-item]
        "target_mask": torch.stack([x["target_mask"] for x in batch]),  # type: ignore[list-item]
        "e_phys": torch.stack([x["e_phys"] for x in batch]),  # type: ignore[list-item]
        "physical_lower": torch.stack([x["physical_lower"] for x in batch]),  # type: ignore[list-item]
        "physical_upper": torch.stack([x["physical_upper"] for x in batch]),  # type: ignore[list-item]
        "row_ids": torch.stack([x["row_ids"] for x in batch]),  # type: ignore[list-item]
        "BS": [str(x["BS"]) for x in batch],
        "origin_time": [str(x["origin_time"]) for x in batch],
    }
    for i, item in enumerate(batch):
        n = int(item["context_x"].shape[0])  # type: ignore[index]
        out["context_x"][i, :n] = item["context_x"]  # type: ignore[index]
        out["context_y"][i, :n] = item["context_y"]  # type: ignore[index]
        out["context_mask"][i, :n] = item["context_mask"]  # type: ignore[index]
    return out


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PGRNPModel(nn.Module):
    def __init__(self, d_in: int, config: PGRNPConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.context_encoder = MLP(d_in + 1, h, h, config.dropout)
        self.target_encoder = MLP(d_in + h + config.latent_dim, h, h, config.dropout)
        self.z_mu = nn.Linear(h, config.latent_dim)
        self.z_logvar = nn.Linear(h, config.latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.n_heads,
            dim_feedforward=h * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.mu_head = nn.Linear(h, 1)
        self.scale_head = nn.Linear(h, 1)
        self.nu_raw = nn.Parameter(torch.tensor(5.0))

    def _context_summary(self, context_x: torch.Tensor, context_y: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.context_encoder(torch.cat([context_x, context_y.unsqueeze(-1)], dim=-1))
        weights = context_mask.unsqueeze(-1)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def forward(self, batch: Dict[str, torch.Tensor], sample_latent: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self._context_summary(batch["context_x"], batch["context_y"], batch["context_mask"])
        z_mu = self.z_mu(ctx)
        z_logvar = self.z_logvar(ctx).clamp(min=-8.0, max=5.0)
        if self.training and sample_latent:
            z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar)
        else:
            z = z_mu

        bsz, n_tgt, _ = batch["target_x"].shape
        ctx_exp = ctx.unsqueeze(1).expand(bsz, n_tgt, -1)
        z_exp = z.unsqueeze(1).expand(bsz, n_tgt, -1)
        target = torch.cat([batch["target_x"], ctx_exp, z_exp], dim=-1)
        h = self.temporal(self.target_encoder(target))
        mu = self.mu_head(h).squeeze(-1)
        scale = F.softplus(self.scale_head(h).squeeze(-1)) + 1e-4
        return mu, scale, z_mu, z_logvar

    def nu(self) -> torch.Tensor:
        return F.softplus(self.nu_raw) + 2.0


def _kl_standard_normal(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp())


def compute_loss(
    model: PGRNPModel,
    batch: Dict[str, torch.Tensor],
    config: PGRNPConfig,
    residual_mean: float,
    residual_std: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mu, scale, z_mu, z_logvar = model(batch)
    target_y = batch["target_y"]
    mask = batch["target_mask"]
    denom = mask.sum().clamp(min=1.0)

    if config.likelihood == "gaussian":
        nll = 0.5 * (torch.log(scale.pow(2)) + ((target_y - mu) / scale).pow(2))
    else:
        nll = -torch.distributions.StudentT(df=model.nu(), loc=mu, scale=scale).log_prob(target_y)
    nll = (nll * mask).sum() / denom
    kl = _kl_standard_normal(z_mu, z_logvar)

    mu_raw = mu * residual_std + residual_mean
    energy_pred = batch["e_phys"] + mu_raw
    bound = (F.relu(batch["physical_lower"] - energy_pred).pow(2) + F.relu(energy_pred - batch["physical_upper"]).pow(2))
    bound = (bound * mask).sum() / denom / (residual_std**2 + 1e-8)

    pair_mask = mask[:, 1:] * mask[:, :-1]
    smooth = ((mu_raw[:, 1:] - mu_raw[:, :-1]).pow(2) * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
    smooth = smooth / (residual_std**2 + 1e-8)
    total = nll + config.lambda_kl * kl + config.lambda_bound * bound + config.lambda_smooth * smooth
    return total, {"total": float(total.detach().cpu()), "nll": float(nll.detach().cpu()), "kl": float(kl.detach().cpu())}


def to_device(batch: Dict[str, torch.Tensor | List[str]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def make_loaders(
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: Dict[str, List[str]],
    config: PGRNPConfig,
) -> Tuple[Dict[str, DataLoader], StandardScaler, float, float]:
    train_df = df[df["BS"].astype(str).isin(splits["train"])].copy()
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32))
    residual_mean = float(train_df["residual_true"].mean())
    residual_std = float(train_df["residual_true"].std(ddof=0))
    datasets = {
        name: ResidualTrajectoryDataset(df, feature_cols, bs, scaler, residual_mean, residual_std, config, train_mode=(name == "train"))
        for name, bs in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True, collate_fn=collate_trajectories),
        "calibration": DataLoader(datasets["calibration"], batch_size=config.batch_size, shuffle=False, collate_fn=collate_trajectories),
        "test": DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, collate_fn=collate_trajectories),
    }
    return loaders, scaler, residual_mean, max(residual_std, 1e-6)


def train_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: Dict[str, List[str]],
    config: PGRNPConfig,
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> Tuple[PGRNPModel, StandardScaler, Dict[str, float], Dict[str, List[float]]]:
    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, scaler, residual_mean, residual_std = make_loaders(df, feature_cols, splits, config)
    model = PGRNPModel(d_in=len(feature_cols), config=config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: Dict[str, List[float]] = {"train": [], "calibration": []}
    best_state = None
    best_val = float("inf")
    patience = 0

    for _ in range(config.epochs):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            optim.zero_grad()
            loss, _ = compute_loss(model, to_device(batch, device), config, residual_mean, residual_std)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loaders["calibration"]:
                loss, _ = compute_loss(model, to_device(batch, device), config, residual_mean, residual_std)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["train"].append(train_loss)
        history["calibration"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pg_rnp_model.pt")
    torch.save({"mean": residual_mean, "std": residual_std}, output_dir / "residual_scaler.pt")
    (output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, scaler, {"mean": residual_mean, "std": residual_std, "best_calibration_loss": best_val}, history


@torch.no_grad()
def predict_rows(
    model: PGRNPModel,
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: Dict[str, List[str]],
    scaler: StandardScaler,
    residual_mean: float,
    residual_std: float,
    config: PGRNPConfig,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    frames = []
    for split_name, bs in splits.items():
        ds = ResidualTrajectoryDataset(df, feature_cols, bs, scaler, residual_mean, residual_std, config, train_mode=False)
        loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_trajectories)
        rows = []
        for batch in loader:
            mu, scale, _, _ = model(to_device(batch, device), sample_latent=False)
            mu_raw = mu.cpu().numpy() * residual_std + residual_mean
            sigma_raw = scale.cpu().numpy() * residual_std
            row_ids = batch["row_ids"].numpy()  # type: ignore[union-attr]
            mask = batch["target_mask"].numpy()  # type: ignore[union-attr]
            for i in range(row_ids.shape[0]):
                for h in range(24):
                    rid = int(row_ids[i, h])
                    if rid < 0 or mask[i, h] <= 0:
                        continue
                    rows.append({"row_id": rid, "split": split_name, "residual_mu": float(mu_raw[i, h]), "residual_sigma": float(max(sigma_raw[i, h], 1e-6))})
        frames.append(pd.DataFrame(rows))

    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    base = df.reset_index(drop=True).reset_index().rename(columns={"index": "row_id"})
    out = base.merge(pred, on="row_id", how="inner")
    out["energy_pred"] = out["E_phys_hat"] + out["residual_mu"]
    out["model_name"] = "A5_PG-RNP_StudentT_PhysicsBound"
    return out
