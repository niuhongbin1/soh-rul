import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -----------------------------
# Utils: data loading & SOH
# -----------------------------
def load_csv_2cols(path):
    df = pd.read_csv(path).iloc[:, :2].copy()
    df.columns = ["time", "state"]
    # 以首值近似额定容量 C0（若你有确切 C0，可替换为常量）
    C0 = float(df["state"].iloc[0])
    df["soh"] = df["state"] / (C0 + 1e-12)
    return df, C0

# -----------------------------
# Detect capacity regeneration (paper: ΔSOH > 0.2%)
# And define regeneration interval until SOH < pre-regeneration SOH
# -----------------------------
@dataclass
class RegenInterval:
    start: int
    end: int  # inclusive

def find_regen_intervals(soh, thr=0.002):
    """Return list of RegenInterval per paper's definition."""
    intervals = []
    n = len(soh)
    k = 1
    while k < n:
        delta = soh[k] - soh[k - 1]
        if delta > thr:
            # regeneration detected at k
            pre_level = soh[k - 1]
            start = k
            j = k + 1
            # interval ends when SOH falls below pre-regeneration level
            while j < n and soh[j] >= pre_level:
                j += 1
            end = min(j, n - 1)
            intervals.append(RegenInterval(start, end))
            k = end + 1
        else:
            k += 1
    return intervals

def in_any_interval(idx, intervals):
    for it in intervals:
        if it.start <= idx <= it.end:
            return True
    return False

# -----------------------------
# Grey models: GM(1,1) for normal, NDGM(1,1) for regeneration
# 简化且工程稳健的实现（与论文思想一致：正常段拟指数趋势，非齐次离散灰模应对再生段的非均匀性）
# -----------------------------
@dataclass
class GMParams:
    a: float
    b: float
    x0_1: float  # x^(0)(1)

def fit_gm(x0):
    """
    GM(1,1) 标准最小二乘估计:
      x^(1) 累加生成，z^(1)(k) = 0.5*(x^(1)(k) + x^(1)(k-1))
      x^(0)(k) + a*z^(1)(k) = b
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    if len(x0) < 3:
        # 极短序列，退化处理
        return GMParams(a=0.0, b=x0.mean(), x0_1=float(x0[0]))
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z1, np.ones_like(z1)])
    Y = x0[1:]
    # [a, b]^T = (B^T B)^{-1} B^T Y
    ata = B.T @ B
    atb = B.T @ Y
    a, b = np.linalg.lstsq(ata, atb, rcond=None)[0]
    return GMParams(a=a, b=b, x0_1=float(x0[0]))

def predict_gm_one_step(p: GMParams, last_x0, k):
    """
    用 GM(1,1) 的时间响应推导一阶还原，做一步预测。
    这里采用常见公式：x^(1)(k) = (x0_1 - b/a) * exp(-a*(k-1)) + b/a
    x^(0)(k) = x^(1)(k) - x^(1)(k-1)
    输入的 k 为 1-based 计数（与公式一致），last_x0 仅用于极端保护。
    """
    a, b, x01 = p.a, p.b, p.x0_1
    if abs(a) < 1e-12:
        return float(last_x0)  # 退化为常值
    def x1(t):
        return (x01 - b / a) * np.exp(-a * (t - 1)) + b / a
    k_int = int(max(2, k))
    x1k = x1(k_int)
    x1k_1 = x1(k_int - 1)
    x0k = x1k - x1k_1
    # 数值保护
    if not np.isfinite(x0k):
        x0k = float(last_x0)
    return float(x0k)

@dataclass
class NDGMParams:
    a: float
    b: float
    c: float
    x0_1: float

def fit_ndgm(x0):
    """
    非齐次离散灰色模型（简化版 NDGM(1,1)）：
       x^(0)(k) + a*z^(1)(k) = b*k + c
    用最小二乘求 a,b,c。
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    if len(x0) < 4:
        # 样本过短时退化为 GM
        gp = fit_gm(x0)
        return NDGMParams(a=gp.a, b=0.0, c=gp.b, x0_1=float(x0[0]))
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])        # k = 2..n
    kvec = np.arange(2, len(x0) + 1, dtype=float)
    Y = x0[1:]
    B = np.column_stack([z1, kvec, np.ones_like(kvec)])
    # Solve: Y = [-a, b, c]·[z1, k, 1]^T  => a取反
    theta, *_ = np.linalg.lstsq(B, Y, rcond=None)
    a_hat, b_hat, c_hat = -theta[0], theta[1], theta[2]
    return NDGMParams(a=float(a_hat), b=float(b_hat), c=float(c_hat), x0_1=float(x0[0]))

def predict_ndgm_one_step(p: NDGMParams, x_hist):
    """
    一步预测：
      x^(0)(k) = -a*z^(1)(k) + b*k + c
      其中 z^(1)(k) = 0.5*(x^(1)(k) + x^(1)(k-1))，x^(1) 为累加序列。
    """
    x_hist = np.asarray(x_hist, dtype=float).ravel()
    n = len(x_hist) + 1  # 预测第 n 点（1-based）
    x1 = np.cumsum(x_hist)
    z1k = 0.5 * (x1[-1] + (x1[-2] if len(x1) >= 2 else x1[-1]))
    x0k = -p.a * z1k + p.b * n + p.c
    if not np.isfinite(x0k):
        x0k = float(x_hist[-1])
    return float(x0k)

# -----------------------------
# EnKF (scalar state, direct observation)
# -----------------------------
class EnKF1D:
    def __init__(self, N=100, q=1e-5, r=1e-4, seed=42):
        self.N = N
        self.q = q  # process noise var
        self.r = r  # obs noise var
        self.rng = np.random.default_rng(seed)

    def step(self, prior_func, x_ens, y_obs):
        """
        prior_func: f(x_prev) -> x_prior (model one-step evolution)
        x_ens: ensemble at k-1 (shape [N])
        y_obs: scalar observation at k
        return: posterior mean scalar, and new ensemble
        """
        # Predict
        x_pred = np.array([prior_func(x) for x in x_ens], dtype=float)
        x_pred += self.rng.normal(0.0, np.sqrt(self.q), size=self.N)

        # Observation operator H = 1, observation ensemble
        y_pred = x_pred + self.rng.normal(0.0, np.sqrt(self.r), size=self.N)
        # Kalman gain (ensemble form)
        x_mean = x_pred.mean()
        y_mean = y_pred.mean()
        Pxy = ((x_pred - x_mean) * (y_pred - y_mean)).sum() / (self.N - 1 + 1e-9)
        Pyy = ((y_pred - y_mean) ** 2).sum() / (self.N - 1 + 1e-9) + self.r
        K = Pxy / (Pyy + 1e-12)

        # Update each member with perturbed obs
        y_tilde = y_obs + self.rng.normal(0.0, np.sqrt(self.r), size=self.N)
        x_upd = x_pred + K * (y_tilde - y_pred)
        return float(x_upd.mean()), x_upd

# -----------------------------
# Rolling forecast with Hybrid Grey + EnKF
# -----------------------------
def hybrid_grey_enkf_forecast(time_raw, state_raw, start_idx=120,
                              enkf_N=200, q=1e-5, r=1e-4, ema_alpha=0.0):
    """
    输入：原始 time/state（不做归一化），从 start_idx 开始滚动预测到末尾。
    训练：仅用 [0, start_idx) 的观测拟合灰模；每步用 EnKF 利用真实 y 进行在线校正。
    返回：pred_series（与 state_raw 等长），以及逐点相对误差。
    """
    n = len(state_raw)
    # SOH 用于再生检测
    C0 = float(state_raw[0])
    soh = state_raw / (C0 + 1e-12)
    regens = find_regen_intervals(soh, thr=0.002)  # 0.2%
    # 初始模型：在训练段分别拟合 GM 与 NDGM（用全训练段，后续按区段选模型）
    train_x = state_raw[:start_idx].astype(float)

    # 为避免拟合偏差，给两个模型都拟合参数
    gm_params = fit_gm(train_x)
    ndgm_params = fit_ndgm(train_x)

    # 初始化 EnKF
    enkf = EnKF1D(N=enkf_N, q=q, r=r)
    # 初始集合从训练段末值附近取
    x0 = train_x[-1]
    x_ens = x0 + np.random.normal(0.0, np.std(train_x[-min(20, len(train_x)):]) + 1e-6, size=enkf_N)

    # 预测数组
    pred = state_raw.copy().astype(float)
    # 历史缓冲（原始尺度）用于 NDGM
    hist = list(train_x)

    # 从 start_idx 开始滚动
    for k in range(start_idx, n):
        # 选择模型：若 k 落在再生区间则用 NDGM，否则用 GM
        use_ndgm = in_any_interval(k, regens)

        if use_ndgm:
            def f_prior(x_prev):
                # NDGM 基于历史序列；为与集合一致，用最近值替换末项形成临时历史
                tmp_hist = np.array(hist[-max(10, len(hist)) : ], dtype=float)
                # 将末项替换为当前集合成员，作为 x_{k-1}
                if len(tmp_hist) >= 1:
                    tmp_hist[-1] = x_prev
                return predict_ndgm_one_step(ndgm_params, tmp_hist)
        else:
            def f_prior(x_prev):
                # GM 一步预测：k 使用 1-based 索引；这里用已走到的位置
                k1 = len(hist) + 1  # 预测下一个的“时刻”
                return predict_gm_one_step(gm_params, x_prev, k1)

        # EnKF 更新（使用真实观测 state_raw[k]）
        y_obs = float(state_raw[k])
        x_mean, x_ens = enkf.step(f_prior, x_ens, y_obs)

        # 可选 EMA 平滑（抑制高频抖动）
        if ema_alpha > 0 and k > start_idx:
            x_mean = ema_alpha * pred[k - 1] + (1 - ema_alpha) * x_mean

        pred[k] = x_mean
        # 推进历史
        hist.append(x_mean)

    rel_err = np.abs(pred - state_raw) / (np.abs(state_raw) + 1e-12)
    return pred, rel_err, regens

# -----------------------------
# Plot helpers
# -----------------------------
def plot_series_with_error(time_raw, y_true, y_pred, start_idx=120, title="Hybrid Grey + EnKF"):
    rel = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12) * 100.0
    over = (rel[start_idx:] > 3.0).mean() * 100.0
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(time_raw, y_true, label="True")
    ax[0].plot(time_raw, y_pred, "--", label="Pred")
    ax[0].axvline(time_raw[start_idx], color="gray", ls=":")
    ax[0].set_ylabel("State/Capacity")
    ax[0].set_title(f"{title} | >3% ratio(after {start_idx}): {over:.2f}%")
    ax[0].legend()
    ax[1].plot(time_raw[start_idx:], rel[start_idx:])
    ax[1].axhline(3.0, color="gray", ls="--")
    ax[1].set_xlabel("Time (CSV col 1)")
    ax[1].set_ylabel("Rel. Error (%)")
    ax[1].set_title("Point-wise Relative Error")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main for three datasets
# -----------------------------
def main():
    files = [
        "./src/p051/data_parse/cleaned_5_max_1.csv",
        "./src/p051/data_parse/cleaned_5_max_2.csv",
        "./src/p051/data_parse/cleaned_5_max_3.csv",
    ]
    start_idx = 120
    for i, fp in enumerate(files, 1):
        df, C0 = load_csv_2cols(fp)
        time_raw = df["time"].values.astype(float)
        state_raw = df["state"].values.astype(float)

        y_pred, rel, regens = hybrid_grey_enkf_forecast(
            time_raw, state_raw, start_idx=start_idx,
            enkf_N=300, q=2e-5, r=5e-5, ema_alpha=0.15
        )
        plot_series_with_error(time_raw, state_raw, y_pred, start_idx=start_idx,
                               title=f"Hybrid Grey + EnKF (Dataset {i})")

        out = pd.DataFrame({"time": time_raw, "state_true": state_raw, "state_pred": y_pred,
                            "rel_err_pct": rel * 100.0})
        out.to_csv(f"forecast_dataset_{i}_grey_enkf.csv", index=False)
        print(f"[Dataset {i}] saved: forecast_dataset_{i}_grey_enkf.csv")
        # 打印再生区间
        if regens:
            print(f"[Dataset {i}] detected regeneration intervals (start,end) indices:",
                  [(it.start, it.end) for it in regens])
        else:
            print(f"[Dataset {i}] no regeneration detected under ΔSOH>0.2%.")

if __name__ == "__main__":
    main()
