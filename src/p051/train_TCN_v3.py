# ===== file: forecast_tcn_xpu_v3.py =====
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def load_and_normalize_csvs(paths):
    data_list, t_scalers, s_scalers = [], [], []
    for p in paths:
        df = pd.read_csv(p).iloc[:, :2].copy()
        df.columns = ["time","state"]
        tsc, ssc = MinMaxScaler(), MinMaxScaler()
        df["time"]  = tsc.fit_transform(df["time"].values.reshape(-1,1)).ravel()
        df["state"] = ssc.fit_transform(df["state"].values.reshape(-1,1)).ravel()
        data_list.append(df); t_scalers.append(tsc); s_scalers.append(ssc)
    return data_list, t_scalers, s_scalers

def make_windows(df, in_len=200, out_len=10, step=1):
    vals = df[["time","state"]].values
    X,Y = [],[]
    for s in range(0, len(vals)-in_len-out_len+1, step):
        seg = vals[s:s+in_len+out_len]
        X.append(seg[:in_len].T)     # (2, in_len)
        Y.append(seg[in_len:,1])     # (out_len,)
    return np.array(X, np.float32), np.array(Y, np.float32)

def build_dataset_all(data_list, in_len=200, out_len=10, step=1):
    Xs,Ys = [],[]
    for df in data_list:
        X,Y = make_windows(df, in_len, out_len, step)
        if len(X): Xs.append(X); Ys.append(Y)
    return np.concatenate(Xs,0), np.concatenate(Ys,0)

class SeqDS(Dataset):
    def __init__(self, X,Y): self.X=torch.tensor(X); self.Y=torch.tensor(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.Y[i]

class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp=chomp
    def forward(self, x): return x[:, :, :-self.chomp].contiguous()

class TBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dil, dropout):
        super().__init__()
        pad=(ks-1)*dil
        self.c1 = nn.utils.weight_norm(nn.Conv1d(in_ch,out_ch,ks,padding=pad,dilation=dil))
        self.chomp1=Chomp1d(pad); self.relu1=nn.ReLU(); self.bn1=nn.BatchNorm1d(out_ch)
        self.c2 = nn.utils.weight_norm(nn.Conv1d(out_ch,out_ch,ks,padding=pad,dilation=dil))
        self.chomp2=Chomp1d(pad); self.relu2=nn.ReLU(); self.dropout=nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        out=self.c1(x); out=self.chomp1(out); out=self.relu1(out); out=self.bn1(out)
        out=self.c2(out); out=self.chomp2(out); out=self.relu2(out); out=self.dropout(out)
        out=out + self.down(x); return self.relu(out)

class TCN(nn.Module):
    def __init__(self, in_ch=2, levels=6, channels=64, ks=3, dropout=0.1, out_len=10):
        super().__init__()
        layers=[]; ch=in_ch
        for i in range(levels):
            layers.append(TBlock(ch, channels, ks, 2**i, dropout)); ch=channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(channels,128), nn.ReLU(), nn.Linear(128,out_len))
    def forward(self, x):
        feat=self.tcn(x); return self.head(feat)  # (B,out_len)

def train(model, tr_loader, va_loader, device, epochs=50, lr=1e-3):
    crit = nn.SmoothL1Loss()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    best=float("inf"); best_state=None; patience=8
    for ep in range(1,epochs+1):
        model.train(); tr=0.0; n=0
        for xb,yb in tr_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss=crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); tr += loss.item()*xb.size(0); n += xb.size(0)
        tr/=n
        model.eval(); va=0.0; m=0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb,yb = xb.to(device), yb.to(device)
                va += crit(model(xb), yb).item()*xb.size(0); m+=xb.size(0)
        va/=m; sched.step(va)
        print(f"Epoch {ep:>3d} | Train {tr:.6f} | Val {va:.6f}")
        if va+1e-8<best: best=va; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience=8
        else:
            patience-=1
            if patience==0: print("Early stopping."); break
    if best_state: model.load_state_dict(best_state)

@torch.no_grad()
def rolling_forecast(model, df_norm, s_scaler, device, in_len=200, out_len=10, start_idx=120):
    t = df_norm["time"].values; s = df_norm["state"].values.copy()
    pred = s.copy(); pos = start_idx
    while pos < len(df_norm):
        if pos-in_len<0: pos+=1; continue
        x = np.stack([t[pos-in_len:pos], pred[pos-in_len:pos]], axis=0)
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)  # (1,2,in_len)
        out = model(x).squeeze(0).detach().cpu().numpy()                      # (out_len,)
        horizon = min(out_len, len(df_norm)-pos)
        pred[pos:pos+horizon] = out[:horizon]; pos += horizon
    true_raw = s_scaler.inverse_transform(df_norm["state"].values.reshape(-1,1)).ravel()
    pred_raw = s_scaler.inverse_transform(pred.reshape(-1,1)).ravel()
    return true_raw, pred_raw

def plot_with_error(raw_time, y_true, y_pred, title, start_idx=120):
    err = np.abs(y_pred - y_true); mask = np.arange(len(raw_time))>=start_idx
    fig, ax = plt.subplots(2,1, figsize=(12,7), sharex=True)
    ax[0].plot(raw_time, y_true, label="True"); ax[0].plot(raw_time, y_pred, "--", label="Pred")
    ax[0].set_title(title); ax[0].set_ylabel("State"); ax[0].legend()
    ax[1].plot(raw_time[mask], err[mask]); ax[1].set_xlabel("Time"); ax[1].set_ylabel("|Error|")
    ax[1].set_title("Point-wise Absolute Error"); plt.tight_layout(); plt.show()

def main():
    files =["./src/p051/data_parse/cleaned_5_max_1.csv","./src/p051/data_parse/cleaned_5_max_2.csv","./src/p051/data_parse/cleaned_5_max_3.csv"]
    in_len,out_len,step = 200,10,1
    batch,epochs,lr = 256,50,1e-3
    start_idx = 120  # <<< 改为从第120点开始
    device=get_device(); print("Using device:", device)

    dfs, _, sscs = load_and_normalize_csvs(files)
    X,Y = build_dataset_all(dfs, in_len, out_len, step); ds=SeqDS(X,Y)
    n_val=max(1,int(len(ds)*0.1)); n_tr=len(ds)-n_val
    tr,va = random_split(ds,[n_tr,n_val],generator=torch.Generator().manual_seed(42))
    tr_loader=DataLoader(tr,batch_size=batch,shuffle=True); va_loader=DataLoader(va,batch_size=batch,shuffle=False)

    model=TCN(in_ch=2, levels=6, channels=64, ks=3, dropout=0.1, out_len=out_len).to(device)
    train(model,tr_loader,va_loader,device,epochs,lr)

    for i,(dfn,ssc) in enumerate(zip(dfs, sscs),1):
        raw_time = pd.read_csv(files[i-1]).iloc[:,0].values
        y_true,y_pred = rolling_forecast(model, dfn, ssc, device, in_len=in_len, out_len=out_len, start_idx=start_idx)
        plot_with_error(raw_time, y_true, y_pred, f"TCN v3 - Dataset {i} (roll from {start_idx})", start_idx=start_idx)
        pd.DataFrame({"time":raw_time,"state_true":y_true,"state_pred":y_pred}).to_csv(
            f"forecast_dataset_{i}_tcn_v3.csv", index=False
        )
        print(f"Saved: forecast_dataset_{i}_tcn_v3.csv")

if __name__=="__main__": main()
