from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..data import load_features_parquet
from ..models import KANModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    return ap.parse_args()


def standardize_cols(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(eps)
    return (x - mean) / std, mean, std


def main() -> None:
    # -------- config --------
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    train_cfg = cfg.get("train", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    # Harden types
    lr: float = float(train_cfg.get("lr", 1e-3))
    epochs: int = int(train_cfg.get("epochs", 25))
    batch_size: int = int(train_cfg.get("batch_size", 256))
    seed: int = int(train_cfg.get("seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    use_cuda = (train_cfg.get("device", "cpu") == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -------- data --------
    features_path = data_cfg["features_file"]
    df = load_features_parquet(features_path)

    # Coerce to numeric, handle inf/NaN, drop fully-empty cols, fill the rest
    df_num = df.apply(pd.to_numeric, errors="coerce")
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    df_num = df_num.dropna(axis=1, how="all")
    df_num = df_num.ffill().bfill().fillna(0)

    if df_num.shape[1] < 2:
        raise ValueError("Not enough numeric columns after coercion to build (X, y).")

    # Choose target column
    target_from_cfg = data_cfg.get("target_col")
    if target_from_cfg and target_from_cfg in df_num.columns:
        y_col = target_from_cfg
    else:
        candidates = [c for c in df_num.columns if c.lower() in {"target", "y", "label"}]
        y_col = candidates[0] if candidates else df_num.columns[-1]

    X_df = df_num.drop(columns=[y_col], errors="ignore").select_dtypes(include=[np.number])
    y_series = df_num[y_col]

    if X_df.empty:
        raise ValueError("Feature frame X is empty after dropping target / non-numeric cols.")

    # Tensors (float32)
    X = torch.tensor(X_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    # keep y raw (we’ll branch below)
    y_raw = torch.tensor(pd.to_numeric(y_series, errors="coerce").to_numpy(), dtype=torch.float32).view(-1, 1)

    # Detect binary classification (two unique values after coercion)
    y_unique = torch.unique(y_raw[~torch.isnan(y_raw)])
    is_binary = (y_unique.numel() == 2)

    # Standardize X
    X, x_mean, x_std = standardize_cols(X)

    # If regression: standardize y; If binary: map to {0,1} and don’t scale y
    if is_binary:
        # Map the smaller value to 0.0, larger to 1.0
        sorted_vals, _ = torch.sort(y_unique)
        y = torch.where(torch.isclose(y_raw, sorted_vals[1]), torch.tensor(1.0), torch.tensor(0.0)).view(-1, 1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        out_dim = 1
        task = "binary"
    else:
        # Regression
        y_mean = y_raw.mean()
        y_std = y_raw.std().clamp_min(1e-6)
        y = (y_raw - y_mean) / y_std
        loss_fn = torch.nn.MSELoss()
        out_dim = 1
        task = "regression"

    # Move to device
    X = X.to(device)
    y = y.to(device)

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[info] Task={task}  target: {y_col} | X: {tuple(X.shape)} | y: {tuple(y.shape)} | device: {device} ({dev_name})")
    print(f"[debug] lr={lr} epochs={epochs} batch_size={batch_size} | X z-scored | y {'bin' if is_binary else 'z-scored'}")

    # -------- dataset / split / loader --------
    ds = TensorDataset(X, y)
    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # -------- model / optim --------
    model = KANModel(
        in_dim=X.shape[1],
        out_dim=out_dim,
        hidden_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 64)),
        depth=int((cfg.get("model", {}) or {}).get("depth", 3)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # -------- train loop --------
    best_val = None
    model.train()
    for epoch in range(epochs):
        # train
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / n_train

        # val
        model.eval()
        with torch.no_grad():
            total_v = 0.0
            for xb, yb in val_loader:
                pred = model(xb)
                total_v += loss_fn(pred, yb).item() * xb.size(0)
            val_loss = total_v / n_val
        model.train()

        if best_val is None or val_loss < best_val:
            best_val = val_loss

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:02d}  train_loss {train_loss:.6f}  val_loss {val_loss:.6f}  best {best_val:.6f}")


if __name__ == "__main__":
    main()
