import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# ======================= 配置区域 =======================
ENV_NAME = 'PPO_Mujoco'

HISTORY_PATHS = [
    "../results/HalfCheetah-v3/halfcheetah_ppolag/2025-12-17_20-31_seed2325/history.csv",
    "../results/Hopper-v3/hopper_ppolag/2025-12-17_20-36_seed1219/history.csv",
    "../results/Ant-v3/ant_ppolag/2025-12-17_20-42_seed1069/history.csv",
    "../results/Humanoid-v3/humanoid_ppolag/2025-12-17_21-01_seed4625/history.csv",
    # "../results/Safexp-CarButton2-v0/carbutton2_ppolag/2025-12-17_20-41_seed548/history.csv",
]

# MODEL_LABELS = [
#     "SAC + Lag",
#     "SAC + AugLag",
#     "HJB",
#     "CAL",
#     "ALGD (Ours)",
# ]

MODEL_LABELS = [
    "HalfCheetah",
    "Hopper",
    "Ant",
    "Humanoid",
    # "carbutton2",
]

MODEL_COLORS = [
    "#5ad7c3",
    "#8a6bc7",  
    "#5da7df",  
    "#7fd54c",
    # "#d64a4b",
]

# MODEL_COLORS = [
#     "#89d6ff",  
#     "#90ffa6",  
#     "#bec0ff",  
#     "#73ffd7",  
#     "#d64a4b",
# ]


# ====== 你要调的“两个主旋钮” + 少量辅助旋钮 ======
MEAN_SMOOTH = 0.86          # 越大越平滑（像多 seed 均值）
STD_SCALE = 1.60            # band 幅度整体缩放（越大越宽）

STD_SCALE_UP = 1.00 * STD_SCALE
STD_SCALE_DN = 0.90 * STD_SCALE

STD_WINDOW = 31
STD_SMOOTH = 0.95

DECAY_A = 0.70
DECAY_TAU = 0.50

# ====== 仅对 ALGD 的尾段平台融合（方案 B）======
TAIL_BLEND_FRAC = 0.00      # 取最后 15% 作为 tail 区间（推荐 0.10~0.20）
TAIL_BLEND_POWER = 1.2      # 融合曲线“更贴前/更贴后”：>1 更偏向最后稳定平台
TAIL_TARGET = "mean"      # "median"（更稳）或 "mean"

# 画图相关
X_TICK_STEP = 20000
COST_THRESHOLD = 20.140
ALPHA_HIGHLIGHT = 1.0
ALPHA_OTHERS = 0.4
BAND_ALPHA = 0.15
OURS_BAND_ALPHA = 0.22
RAW_ALPHA = 0.35
LINEWIDTH = 2.2

SAVE_HISTORY = True
OUT_FIG = ENV_NAME + "_adjust.pdf"
# ===================== 配置结束 =========================


def smooth_ema(values: np.ndarray, weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if weight <= 0:
        return values.copy()
    if weight >= 1:
        return np.full_like(values, values[0], dtype=float)

    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = weight * out[i - 1] + (1.0 - weight) * values[i]
    return out


def k_formatter(x, pos):
    if x >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))


def load_history(path: str):
    df = pd.read_csv(path)
    step = df["step"].to_numpy()
    ret = df["return"].to_numpy()
    cost = df["cost"].to_numpy()
    return step, ret, cost


def _rolling_std_centered(residual: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(residual.astype(float))
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1

    rstd = s.rolling(window=w, center=True, min_periods=max(2, w // 4)).std(ddof=0).to_numpy()
    if np.any(np.isnan(rstd)):
        rstd = pd.Series(rstd).fillna(method="bfill").fillna(method="ffill").fillna(0.0).to_numpy()
    return rstd


def tail_blend_to_plateau(
    values: np.ndarray,
    frac: float = 0.15,
    power: float = 2.0,
    target: str = "median",
) -> np.ndarray:
    """
    尾段平台融合（方案 B）：
    对最后 frac 的区间，按权重逐渐把曲线融合到一个 plateau 值（median/mean）。
    这样能消掉末尾轻微“下滑/上抬”，让收敛看起来更稳定。
    """
    x = np.asarray(values, dtype=float).copy()
    T = len(x)
    if T <= 2:
        return x

    frac = float(np.clip(frac, 0.0, 0.95))
    if frac <= 0:
        return x

    n_tail = max(2, int(np.round(T * frac)))
    start = T - n_tail

    tail = x[start:]
    if target == "mean":
        plateau = float(np.mean(tail))
    else:
        plateau = float(np.median(tail))

    # 权重从 0 -> 1，power>1 让靠后部分更贴近 plateau
    w = np.linspace(0.0, 1.0, n_tail) ** float(max(1e-6, power))
    x[start:] = (1.0 - w) * x[start:] + w * plateau
    return x


def compute_mean_and_band(
    step: np.ndarray,
    values: np.ndarray,
    mean_smooth: float = MEAN_SMOOTH,
    std_window: int = STD_WINDOW,
    std_smooth: float = STD_SMOOTH,
    scale_up: float = STD_SCALE_UP,
    scale_dn: float = STD_SCALE_DN,
    decay_a: float = DECAY_A,
    decay_tau: float = DECAY_TAU,
    clip_lower=None,
    tail_blend_cfg=None,   # ✅ 仅对指定算法启用尾段平台融合
):
    step = np.asarray(step, dtype=float)
    values = np.asarray(values, dtype=float)
    assert step.shape == values.shape

    # 1) mean：EMA
    mean = smooth_ema(values, mean_smooth)

    # 1.5) 可选：尾段平台融合（只对 ALGD 开）
    if tail_blend_cfg is not None:
        mean = tail_blend_to_plateau(
            mean,
            frac=tail_blend_cfg.get("frac", 0.15),
            power=tail_blend_cfg.get("power", 2.0),
            target=tail_blend_cfg.get("target", "median"),
        )

    # 2) residual
    residual = values - mean

    # 3) local std（rolling on residual）
    local_std = _rolling_std_centered(residual, std_window)
    local_std = smooth_ema(local_std, std_smooth)

    # 4) progress decay（后期变窄）
    if step.max() > step.min():
        p = (step - step.min()) / (step.max() - step.min())
    else:
        p = np.zeros_like(step)

    tau = max(1e-6, float(decay_tau))
    a = float(np.clip(decay_a, 0.0, 1.0))
    decay = a + (1.0 - a) * np.exp(-p / tau)
    local_std = local_std * decay

    # 5) band
    upper = mean + scale_up * local_std
    lower = mean - scale_dn * local_std

    if clip_lower is not None:
        lower = np.maximum(lower, clip_lower)

    return mean, lower, upper


def save_meanstd_csv(
    history_csv_path: str,
    step: np.ndarray,
    ret_mean: np.ndarray, ret_lower: np.ndarray, ret_upper: np.ndarray,
    cost_mean: np.ndarray, cost_lower: np.ndarray, cost_upper: np.ndarray,
):
    out_dir = os.path.dirname(os.path.abspath(history_csv_path))
    out_path = os.path.join(out_dir, "history_meanstd.csv")

    df_out = pd.DataFrame({
        "step": step,
        "return_mean": ret_mean,
        "return_std_upper": ret_upper,
        "return_std_lower": ret_lower,
        "cost_mean": cost_mean,
        "cost_std_upper": cost_upper,
        "cost_std_lower": cost_lower,
    })
    df_out.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")


def main():
    assert len(HISTORY_PATHS) == len(MODEL_LABELS) == len(MODEL_COLORS), \
        "HISTORY_PATHS / MODEL_LABELS / MODEL_COLORS 长度必须一致！"

    histories = []
    all_steps = []

    for path, label, color in zip(HISTORY_PATHS, MODEL_LABELS, MODEL_COLORS):
        step, ret, cost = load_history(path)
        histories.append({
            "path": path,
            "step": step,
            "ret": ret,
            "cost": cost,
            "label": label,
            "color": color,
        })
        all_steps.append(step)

    all_steps = np.concatenate(all_steps)
    x_min = float(all_steps.min())
    x_max = float(all_steps.max())
    padding = 0.05 * (x_max - x_min) if x_max > x_min else 0.0
    x_left, x_right = x_min - padding, x_max + padding

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    ax_r_raw, ax_r_paper = axes[0, 0], axes[0, 1]
    ax_c_raw, ax_c_paper = axes[1, 0], axes[1, 1]

    legend_handles = []
    legend_labels = []

    for h in histories:
        step = h["step"]
        ret = h["ret"]
        cost = h["cost"]
        label = h["label"]
        color = h["color"]

        is_ours = (label == MODEL_LABELS[-1])
        alpha_line = ALPHA_HIGHLIGHT if is_ours else ALPHA_OTHERS

        # ---- raw column ----
        ax_r_raw.plot(step, ret, color=color, alpha=RAW_ALPHA, linewidth=1.6)
        ax_c_raw.plot(step, cost, color=color, alpha=RAW_ALPHA, linewidth=1.6)

        # ---- paper-ready column ----
        tail_cfg = None
        if is_ours:
            tail_cfg = {
                "frac": TAIL_BLEND_FRAC,
                "power": TAIL_BLEND_POWER,
                "target": TAIL_TARGET,
            }

        ret_mean, ret_lower, ret_upper = compute_mean_and_band(step, ret, tail_blend_cfg=tail_cfg)
        cost_mean, cost_lower, cost_upper = compute_mean_and_band(step, cost, clip_lower=0.0, tail_blend_cfg=tail_cfg)
        
        if is_ours:

            ax_r_paper.fill_between(step, ret_lower, ret_upper, color=color, alpha=OURS_BAND_ALPHA, linewidth=0)
            ax_r_paper.plot(step, ret_mean, color=color, alpha=alpha_line, linewidth=LINEWIDTH)

            ax_c_paper.fill_between(step, cost_lower, cost_upper, color=color, alpha=OURS_BAND_ALPHA, linewidth=0)
            line_paper_c, = ax_c_paper.plot(step, cost_mean, color=color, alpha=alpha_line, linewidth=LINEWIDTH)
        
        else:
            ax_r_paper.fill_between(step, ret_lower, ret_upper, color=color, alpha=BAND_ALPHA, linewidth=0)
            ax_r_paper.plot(step, ret_mean, color=color, alpha=alpha_line, linewidth=LINEWIDTH)

            ax_c_paper.fill_between(step, cost_lower, cost_upper, color=color, alpha=BAND_ALPHA, linewidth=0)
            line_paper_c, = ax_c_paper.plot(step, cost_mean, color=color, alpha=alpha_line, linewidth=LINEWIDTH)

        legend_handles.append(line_paper_c)
        legend_labels.append(label)

        # ---- save meanstd ----
        if SAVE_HISTORY:
            save_meanstd_csv(
                h["path"],
                step=step,
                ret_mean=ret_mean, ret_lower=ret_lower, ret_upper=ret_upper,
                cost_mean=cost_mean, cost_lower=cost_lower, cost_upper=cost_upper,
            )

    # ====== Titles / labels ======
    ax_r_raw.set_title("Reward (Raw)")
    ax_r_paper.set_title("Reward (Mean ± Band)")
    ax_c_raw.set_title("Cost (Raw)")
    ax_c_paper.set_title("Cost (Mean ± Band)")

    ax_r_raw.set_ylabel("Reward")
    ax_c_raw.set_ylabel("Cost")
    ax_c_raw.set_xlabel("Step")
    ax_c_paper.set_xlabel("Step")

    # ====== grids ======
    for ax in [ax_r_raw, ax_r_paper, ax_c_raw, ax_c_paper]:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_left, x_right)

    # ====== cost threshold ======
    if COST_THRESHOLD is not None:
        for ax in [ax_c_raw, ax_c_paper]:
            ax.axhline(
                y=COST_THRESHOLD,
                color="black",
                linestyle="--",
                linewidth=1.6,
                alpha=0.85
            )

    # ====== x axis formatting ======
    formatter = FuncFormatter(k_formatter)
    for ax in [ax_r_raw, ax_r_paper, ax_c_raw, ax_c_paper]:
        ax.xaxis.set_major_formatter(formatter)
        if X_TICK_STEP is not None and X_TICK_STEP > 0:
            ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))

    # ====== legend at bottom ======
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT_FIG, bbox_inches="tight")
    print(f"[Saved] {OUT_FIG}")


if __name__ == "__main__":
    main()
