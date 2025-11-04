import os
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ---------------- Device selection ----------------
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------- Data utils ----------------
def load_and_normalize_csvs(file_paths: List[str]):
    """
    CSV: 2 columns -> [time, state]
    Per-dataset, per-column MinMax normalization.
    """
    data_list, t_scalers, s_scalers = [], [], []
    for p in file_paths:
        df = pd.read_csv(p).iloc[:, :2].copy()
        df.columns = ["time", "state"]
        tsc = MinMaxScaler()
        ssc = MinMaxScaler()
        df["time"] = tsc.fit_transform(df["time"].values.reshape(-1, 1)).ravel()
        df["state"] = ssc.fit_transform(df["state"].values.reshape(-1, 1)).ravel()
        data_list.append(df)
        t_scalers.append(tsc)
        s_scalers.append(ssc)
    return data_list, t_scalers, s_scalers


def make_windows(df: pd.DataFrame, in_len=100, out_len=10, step=1):
    """
    X: (in_len, 2) -> [time, state]
    y: (out_len, 1) -> only state
    """
    vals = df[["time", "state"]].values
    X, Y = [], []
    total = len(vals)
    for s in range(0, total - in_len - out_len + 1, step):
        seg = vals[s : s + in_len + out_len]
        X.append(seg[:in_len])
        Y.append(seg[in_len:, 1:2])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y


def build_dataset_from_all(data_list, in_len=100, out_len=10, step=1):
    Xs, Ys = [], []
    for df in data_list:
        X, Y = make_windows(df, in_len=in_len, out_len=out_len, step=step)
        if len(X) > 0:
            Xs.append(X)
            Ys.append(Y)
    if not Xs:
        raise ValueError("No samples created. Check window lengths vs. CSV length.")
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ---------------- Model ----------------
class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, in_len, 2)
        out, h = self.gru(x)  # h: (num_layers, B, hidden)
        h = self.ln(h)        # stabilize hidden
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, y_prev, h):
        # y_prev: (B, 1, 1), h: (num_layers, B, hidden)
        out, h = self.gru(y_prev, h)  # (B, 1, hidden)
        y = self.fc(out)              # (B, 1, 1)
        return y, h


class Seq2SeqGRU(nn.Module):
    """
    Autoregressive 1-step decoder with teacher forcing support.
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, out_len=10, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(hidden_dim, num_layers, dropout)
        self.out_len = out_len

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        """
        x: (B, in_len, 2)
        y: (B, out_len, 1) ground truth (optional for teacher forcing)
        """
        B = x.size(0)
        h = self.encoder(x)

        # start token = last observed state
        dec_in = x[:, -1:, 1:2]  # (B,1,1)
        outs = []
        for t in range(self.out_len):
            out, h = self.decoder(dec_in, h)  # (B,1,1)
            outs.append(out)
            if (y is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                dec_in = y[:, t:t+1, :]       # teacher forcing
            else:
                dec_in = out                   # autoregressive
        return torch.cat(outs, dim=1)          # (B,out_len,1)


# ---------------- Train & Evaluate ----------------
def train_model(model, train_loader, val_loader, device,
                epochs=30, lr=1e-3, tf_start=0.7, tf_end=0.0, early_stop_patience=6):
    crit = nn.SmoothL1Loss()  # Huber
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_val = float("inf")
    best_state = None
    patience = early_stop_patience

    for ep in range(1, epochs + 1):
        # teacher forcing linearly anneals
        tf_ratio = tf_start + (tf_end - tf_start) * (ep - 1) / max(1, epochs - 1)

        # ---- train ----
        model.train()
        tr_loss, n_tr = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb, yb, teacher_forcing_ratio=tf_ratio)
            loss = crit(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n_tr += xb.size(0)
        tr_loss /= n_tr

        # ---- val ----
        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)  # no teacher forcing
                loss = crit(preds, yb)
                val_loss += loss.item() * xb.size(0)
                n_v += xb.size(0)
        val_loss /= n_v
        scheduler.step(val_loss)

        print(f"Epoch {ep:>3d} | tf={tf_ratio:.2f} | Train {tr_loss:.6f} | Val {val_loss:.6f}")

        # early stopping
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stop_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def rolling_forecast_from_index(model, df_norm: pd.DataFrame, s_scaler: MinMaxScaler,
                                device, in_len=100, out_len=10, start_idx=70):
    """
    从 start_idx 开始，使用前 in_len 个点作为上下文，
    每次预测 1..out_len，递归前进，直到序列末尾。
    返回：raw_time, raw_state_true, raw_state_pred
    """
    time_n = df_norm["time"].values
    state_n = df_norm["state"].values.copy()
    pred_n = state_n.copy()  # 覆盖 start_idx 及之后为预测

    pos = start_idx
    model.eval()
    while pos < len(df_norm):
        # 需要确保有 in_len 历史
        if pos - in_len < 0:
            pos += 1
            continue
        # 构建输入
        x_in = np.stack([time_n[pos - in_len: pos],
                         pred_n[pos - in_len: pos]], axis=1)
        x_in = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)  # (1,in_len,2)

        # 预测本轮最多 out_len 步，但不要越界
        horizon = min(out_len, len(df_norm) - pos)
        preds_block = []
        dec_in = x_in[:, -1:, 1:2]  # last observed state (=pred_n[pos-1])

        h = model.encoder(x_in)  # 手动展开 decoder (避免 forward 里固定长度)
        for _ in range(horizon):
            y, h = model.decoder(dec_in, h)
            preds_block.append(y)
            dec_in = y

        preds_block = torch.cat(preds_block, dim=1).squeeze(0).squeeze(-1).cpu().numpy()
        pred_n[pos: pos + horizon] = preds_block
        pos += horizon

    # 反归一化
    state_true_raw = s_scaler.inverse_transform(state_n.reshape(-1, 1)).ravel()
    state_pred_raw = s_scaler.inverse_transform(pred_n.reshape(-1, 1)).ravel()
    return state_true_raw, state_pred_raw


def plot_compare(raw_time, state_true_raw, state_pred_raw, title):
    plt.figure(figsize=(12, 5))
    plt.plot(raw_time, state_true_raw, label="True")
    plt.plot(raw_time, state_pred_raw, "--", label="Predicted")
    plt.xlabel("Time (CSV column 1)")
    plt.ylabel("State Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------- Main ----------------
def main():
    # ---------- config ----------
    files = [
        "./src/p051/data_parse/cleaned_5_max_1.csv",
        "./src/p051/data_parse/cleaned_5_max_2.csv",
        "./src/p051/data_parse/cleaned_5_max_3.csv",
    ]
    in_len = 100
    out_len = 10
    train_step = 1      # 强烈建议=1，样本更密
    batch_size = 256
    hidden = 128
    layers = 2
    dropout = 0.2
    epochs = 40
    lr = 1e-3
    tf_start = 0.7
    tf_end = 0.0
    val_ratio = 0.1
    start_idx = 70

    device = get_device()
    print("Using device:", device)

    # ---------- data ----------
    dfs_norm, t_scalers, s_scalers = load_and_normalize_csvs(files)
    X, Y = build_dataset_from_all(dfs_norm, in_len=in_len, out_len=out_len, step=train_step)
    dataset = SeqDataset(X, Y)

    # train/val split（按样本随机划分）
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ---------- model ----------
    model = Seq2SeqGRU(input_dim=2, hidden_dim=hidden, num_layers=layers,
                       out_len=out_len, dropout=dropout).to(device)

    # ---------- train ----------
    train_model(model, train_loader, val_loader, device,
                epochs=epochs, lr=lr, tf_start=tf_start, tf_end=tf_end, early_stop_patience=6)

    # ---------- rolling forecast for each dataset ----------
    for i, (dfn, ssc) in enumerate(zip(dfs_norm, s_scalers), start=1):
        # 原始时间用于横轴
        raw_time = pd.read_csv(files[i-1]).iloc[:, 0].values
        state_true_raw, state_pred_raw = rolling_forecast_from_index(
            model, dfn, ssc, device, in_len=in_len, out_len=out_len, start_idx=start_idx
        )
        plot_compare(raw_time, state_true_raw, state_pred_raw,
                     title=f"Dataset {i}: Rolling Forecast from index {start_idx} to end")

        # （可选）保存结果到 CSV
        out = pd.DataFrame({
            "time": raw_time,
            "state_true": state_true_raw,
            "state_pred": state_pred_raw
        })
        out.to_csv(f"forecast_dataset_{i}.csv", index=False)
        print(f"Saved: forecast_dataset_{i}.csv")


if __name__ == "__main__":
    main()
