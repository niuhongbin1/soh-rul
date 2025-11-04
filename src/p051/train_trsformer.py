# ===== file: forecast_transformer_xpu.py =====
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ---------------- Device ----------------
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------- Data ----------------
def load_and_normalize_csvs(file_paths):
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


def make_windows(df, in_len=100, out_len=10, step=1):
    vals = df[["time", "state"]].values
    X, Y = [], []
    for s in range(0, len(vals) - in_len - out_len + 1, step):
        seg = vals[s : s + in_len + out_len]
        X.append(seg[:in_len])           # (in_len, 2)
        Y.append(seg[in_len:, 1:2])      # (out_len, 1)
    X = np.array(X, dtype=np.float32)    # (N, in_len, 2)
    Y = np.array(Y, dtype=np.float32)    # (N, out_len, 1)
    return X, Y


def build_dataset_from_all(data_list, in_len=100, out_len=10, step=1):
    Xs, Ys = [], []
    for df in data_list:
        X, Y = make_windows(df, in_len, out_len, step)
        if len(X) > 0:
            Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ---------------- Positional Encoding ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x):  # x: (S, B, d_model)
        S = x.size(0)
        return x + self.pe[:S]


# ---------------- Transformer Seq2Seq ----------------
class TransSeq2Seq(nn.Module):
    """
    Encoder 输入： (in_len, B, d_model)  <- 线性映射自 (in_len, B, 2)
    Decoder 输入： (out_len, B, d_model) <- 线性映射自 (out_len, B, 1)
    训练：teacher forcing；推理：自回归展开 out_len。
    """
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1, out_len=10):
        super().__init__()
        self.out_len = out_len
        self.src_proj = nn.Linear(2, d_model)
        self.tgt_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.head = nn.Linear(d_model, 1)

    def encode(self, src):  # src: (B, in_len, 2)
        src = self.src_proj(src)                 # (B, in_len, d)
        src = src.transpose(0, 1)               # (in_len, B, d)
        src = self.pos_enc(src)
        memory = self.transformer.encoder(src)  # (in_len, B, d)
        return memory

    def decode(self, tgt, memory):  # tgt: (B, L, 1)
        tgt = self.tgt_proj(tgt).transpose(0, 1)   # (L, B, d)
        tgt = self.pos_enc(tgt)
        # causal mask
        L = tgt.size(0)
        mask = torch.triu(torch.ones(L, L, device=tgt.device), diagonal=1).bool()
        out = self.transformer.decoder(tgt, memory, tgt_mask=mask)   # (L,B,d)
        out = self.head(out).transpose(0, 1)                         # (B,L,1)
        return out

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        """
        src: (B, in_len, 2)
        tgt: (B, out_len, 1)  -> ground truth for teacher forcing
        """
        B = src.size(0)
        memory = self.encode(src)  # (in_len,B,d)

        if self.training and tgt is not None:
            # teacher forcing：把真值整体作为 decoder 输入，但首步用 last observed state 作为起始
            start = src[:, -1:, 1:2]           # (B,1,1)
            tf_in = torch.cat([start, tgt[:, :-1, :]], dim=1)  # (B,out_len,1) shift right
            return self.decode(tf_in, memory)  # (B,out_len,1)

        # inference: auto-regressive
        outs = []
        dec_in = src[:, -1:, 1:2]  # (B,1,1)
        for _ in range(self.out_len):
            y = self.decode(dec_in, memory)[:, -1:, :]  # 仅取最后一步
            outs.append(y)
            dec_in = torch.cat([dec_in, y], dim=1)
        return torch.cat(outs, dim=1)  # (B,out_len,1)


# ---------------- Train / Forecast ----------------
def train(model, tr_loader, va_loader, device, epochs=40, lr=1e-3, tf_ratio=0.7):
    crit = nn.SmoothL1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best, best_state, patience = float("inf"), None, 6
    for ep in range(1, epochs + 1):
        # 线性退火
        tf = tf_ratio * (1 - (ep - 1) / max(1, epochs - 1))
        model.train()
        tr, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, yb, teacher_forcing_ratio=tf)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * xb.size(0); n += xb.size(0)
        tr /= n

        model.eval()
        va, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)  # no teacher forcing
                loss = crit(pred, yb)
                va += loss.item() * xb.size(0); m += xb.size(0)
        va /= m
        scheduler.step(va)
        print(f"Epoch {ep:>3d} | tf={tf:.2f} | Train {tr:.6f} | Val {va:.6f}")

        if va + 1e-8 < best:
            best, best_state, patience = va, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 6
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break
    if best_state: model.load_state_dict(best_state)


@torch.no_grad()
def rolling_forecast(model, df_norm, s_scaler, device, in_len=100, out_len=10, start_idx=70):
    time_n = df_norm["time"].values
    state_n = df_norm["state"].values.copy()
    pred_n = state_n.copy()

    pos = start_idx
    while pos < len(df_norm):
        if pos - in_len < 0:
            pos += 1; continue

        src = np.stack([time_n[pos-in_len:pos], pred_n[pos-in_len:pos]], axis=1)  # (in_len,2)
        src = torch.tensor(src, dtype=torch.float32, device=device).unsqueeze(0)  # (1,in_len,2)

        # 自回归最多 out_len 步，不越界
        horizon = min(out_len, len(df_norm) - pos)
        outs = []
        dec_in = src[:, -1:, 1:2]  # (1,1,1)
        memory = model.encode(src)

        for _ in range(horizon):
            y = model.decode(dec_in, memory)[:, -1:, :]  # last step
            outs.append(y)
            dec_in = torch.cat([dec_in, y], dim=1)

        outs = torch.cat(outs, dim=1).squeeze(0).squeeze(-1).cpu().numpy()
        pred_n[pos:pos+horizon] = outs
        pos += horizon

    state_true = s_scaler.inverse_transform(df_norm["state"].values.reshape(-1,1)).ravel()
    state_pred = s_scaler.inverse_transform(pred_n.reshape(-1,1)).ravel()
    return state_true, state_pred


def plot_with_error(raw_time, true_raw, pred_raw, title, start_idx=70):
    err = pred_raw - true_raw
    abs_err = np.abs(err)
    mask = np.arange(len(raw_time)) >= start_idx

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(raw_time, true_raw, label="True")
    ax[0].plot(raw_time, pred_raw, "--", label="Predicted")
    ax[0].set_ylabel("State Value")
    ax[0].set_title(title)
    ax[0].legend()

    ax[1].plot(raw_time[mask], abs_err[mask])
    ax[1].set_xlabel("Time (CSV column 1)")
    ax[1].set_ylabel("|Error| per point")
    ax[1].set_title("Point-wise Absolute Error")

    plt.tight_layout()
    plt.show()


def main():
    files =["./src/p051/data_parse/cleaned_5_max_1.csv","./src/p051/data_parse/cleaned_5_max_2.csv","./src/p051/data_parse/cleaned_5_max_3.csv"]
    in_len, out_len, step = 100, 10, 1
    batch, epochs, lr = 128, 40, 1e-3
    d_model, nhead, nlayers, dff, dropout = 128, 8, 3, 256, 0.1

    device = get_device()
    print("Using device:", device)

    dfs_norm, _, sscs = load_and_normalize_csvs(files)
    X, Y = build_dataset_from_all(dfs_norm, in_len=in_len, out_len=out_len, step=step)
    ds = SeqDataset(X, Y)

    # split
    val_ratio = 0.1
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    tr_set, va_set = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_set, batch_size=batch, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch, shuffle=False)

    model = TransSeq2Seq(d_model=d_model, nhead=nhead, num_layers=nlayers,
                         dim_feedforward=dff, dropout=dropout, out_len=out_len).to(device)
    train(model, tr_loader, va_loader, device, epochs=epochs, lr=lr, tf_ratio=0.7)

    for i, (dfn, ssc) in enumerate(zip(dfs_norm, sscs), start=1):
        raw_time = pd.read_csv(files[i-1]).iloc[:, 0].values
        true_raw, pred_raw = rolling_forecast(model, dfn, ssc, device, in_len=in_len, out_len=out_len, start_idx=70)
        plot_with_error(raw_time, true_raw, pred_raw, f"Transformer - Dataset {i} (roll from 70)", start_idx=70)
        pd.DataFrame({"time": raw_time, "state_true": true_raw, "state_pred": pred_raw}).to_csv(
            f"forecast_dataset_{i}_transformer.csv", index=False
        )
        print(f"Saved: forecast_dataset_{i}_transformer.csv")


if __name__ == "__main__":
    main()
