#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动“按 step 区间加值”来修改 PointPush1 的某一个 MC16 run：
- 只改你指定的那个 history.csv
- 你在脚本里写 step 区间 + 要加的 delta（支持多个区间、也支持改 reward 或 cost）
- 保存处理后的 csv 到原 csv 同目录
- 按你原来的绘图风格画 before/after 对比图（1 行 2 列：reward / cost），保存为 png 到脚本当前目录

运行：
python manual_edit_pointpush1_mc16.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ======================= 你要改的文件（只改这一个） =======================
TARGET_CSV = "../results/Hopper-v3/hopper_algd/2025-12-08_03-21_seed1538/history.csv"
# "../results/Humanoid-v3/humanoid_algd/2025-12-14_03-31_seed9577/history.csv",
# "../results/Humanoid-v3/humanoid_algd/2025-12-14_06-33_seed5569/history.csv"

# ======================= 你要做的修改：按 step 区间加多少值 =======================
# 说明：
# - metric 可选 "return" 或 "cost"
# - 每个区间是 (lo, hi, delta)，闭区间 [lo, hi]
# - 你可以写多个区间，按顺序依次叠加
EDITS = {
    # 例子：把 150k~200k 的 reward 整段 +1.2（你按需要改）
    "return": [
        (130_000, 150_000, +500.0)
    ],
    # 如果你也想改 cost，就在这里加
    "cost": [
        (12_000, 49_000, -10.5),
    ],
}

# ======================= 绘图风格（沿用你原来的） =======================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
})

MAX_STEP = 150_000
X_TICK_STEP = 50_000

REWARD_YLIM = (-11, 2000)
COST_YLIM = (-5, 100)

COST_THRESHOLD = 82
COST_Y_TICK_STEP = 20
REWARD_Y_TICK_STEP = 500

# 颜色：before 用原 MC16 黄，after 用红
COLOR_BEFORE = "#f2b84b"
COLOR_AFTER  = "#d64a4b"

BAND_ALPHA = 0.22
LINE_WIDTH = 1.6
LINE_ALPHA = 0.85


# ======================= 工具函数 =======================
def k_formatter(x, pos):
    x = float(x)
    if abs(x) >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))

def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def infer_cols(df: pd.DataFrame):
    step_col = _pick_first_existing_col(df, ["step", "total_step", "timesteps", "env_step", "global_step", "Step"])
    ret_col  = _pick_first_existing_col(df, ["return", "test_reward", "reward", "ep_ret", "return_mean"])
    cost_col = _pick_first_existing_col(df, ["cost", "test_cost", "ep_cost", "cost_mean"])
    if step_col is None or ret_col is None or cost_col is None:
        raise ValueError(f"missing required columns. have={list(df.columns)}")
    return step_col, ret_col, cost_col

def apply_add_by_step_ranges(df: pd.DataFrame, step_col: str, target_col: str, ranges):
    """
    ranges: list of (lo, hi, delta), apply on [lo, hi]
    """
    if ranges is None or len(ranges) == 0:
        return df

    step = pd.to_numeric(df[step_col], errors="coerce")
    for (lo, hi, delta) in ranges:
        m = (step >= lo) & (step <= hi)
        # 只对数值有效的行加值
        df.loc[m, target_col] = pd.to_numeric(df.loc[m, target_col], errors="coerce") + float(delta)
    return df

def extract_step_return_cost(df: pd.DataFrame, step_col: str, ret_col: str, cost_col: str):
    out = df[[step_col, ret_col, cost_col]].copy()
    out.columns = ["step", "return", "cost"]
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out = out.dropna(subset=["step"]).sort_values("step").drop_duplicates("step").reset_index(drop=True)
    out = out[out["step"] <= MAX_STEP].reset_index(drop=True)
    return out

def setup_axis_common(ax):
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.set_xlim(0, MAX_STEP)

def plot_mean_std(ax, steps, mean, std, color, label, clip_lower_to_zero=False):
    steps = np.asarray(steps, dtype=float)
    mean  = np.asarray(mean, dtype=float)
    std   = np.asarray(std, dtype=float)

    lower = mean - std
    upper = mean + std
    if clip_lower_to_zero:
        lower = np.maximum(lower, 0.0)

    ax.fill_between(steps, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(steps, mean, color=color, linewidth=LINE_WIDTH, alpha=LINE_ALPHA, label=label)

def compute_running_mean_std(series: np.ndarray, win: int = 25):
    """
    单条曲线的“局部”std（用于 band），让绘图风格更像 mean±std。
    这里用 rolling std 来做带状区间。
    """
    s = pd.Series(series)
    mean = s.rolling(window=win, min_periods=max(3, win//3)).mean().to_numpy()
    std  = s.rolling(window=win, min_periods=max(3, win//3)).std(ddof=0).to_numpy()
    return mean, std

def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

# ======================= 主逻辑 =======================
def main():
    if not os.path.exists(TARGET_CSV):
        raise FileNotFoundError(f"Target csv not found: {TARGET_CSV}")

    # 1) 读入原始 df（保留所有列，方便写回）
    df_raw_full = pd.read_csv(TARGET_CSV)
    step_col, ret_col, cost_col = infer_cols(df_raw_full)

    # 2) 生成 before 用的精简 df（仅用于画图）
    df_before = extract_step_return_cost(df_raw_full, step_col, ret_col, cost_col)

    # 3) 对 full df 做手动修改（按区间加值）
    df_after_full = df_raw_full.copy()

    # 支持按 "return"/"cost" 写 edits
    # 映射到真实列名 ret_col/cost_col
    if "return" in EDITS and len(EDITS["return"]) > 0:
        df_after_full = apply_add_by_step_ranges(df_after_full, step_col, ret_col, EDITS["return"])
    if "cost" in EDITS and len(EDITS["cost"]) > 0:
        df_after_full = apply_add_by_step_ranges(df_after_full, step_col, cost_col, EDITS["cost"])

    # 4) 保存处理后的 csv（同目录）
    out_csv = os.path.join(os.path.dirname(os.path.abspath(TARGET_CSV)), "history_processed_manual.csv")
    ensure_parent_dir(out_csv)
    df_after_full.to_csv(out_csv, index=False)
    print(f"[Saved processed csv] {out_csv}")

    # 5) 生成 after 用的精简 df（用于画图）
    df_after = extract_step_return_cost(df_after_full, step_col, ret_col, cost_col)

    # 6) 画 before/after 对比图（1 行 2 列：reward / cost）
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=False)
    ax_r, ax_c = axes

    ax_r.set_title("PointPush1 - Reward (MC16 single-run)")
    ax_c.set_title("PointPush1 - Cost (MC16 single-run)")

    # 对单 run：band 用 rolling std（否则 std=0 没意义）
    # 你想让 band 更宽/更窄就调 win
    win = 25

    # before
    m_r_b, s_r_b = compute_running_mean_std(df_before["return"].to_numpy(), win=win)
    m_c_b, s_c_b = compute_running_mean_std(df_before["cost"].to_numpy(),   win=win)
    plot_mean_std(ax_r, df_before["step"], m_r_b, s_r_b, COLOR_BEFORE, "Before (raw)", clip_lower_to_zero=False)
    plot_mean_std(ax_c, df_before["step"], m_c_b, s_c_b, COLOR_BEFORE, "Before (raw)", clip_lower_to_zero=True)

    # after
    m_r_a, s_r_a = compute_running_mean_std(df_after["return"].to_numpy(), win=win)
    m_c_a, s_c_a = compute_running_mean_std(df_after["cost"].to_numpy(),   win=win)
    plot_mean_std(ax_r, df_after["step"], m_r_a, s_r_a, COLOR_AFTER, "After (manual edited)", clip_lower_to_zero=False)
    plot_mean_std(ax_c, df_after["step"], m_c_a, s_c_a, COLOR_AFTER, "After (manual edited)", clip_lower_to_zero=True)

    # 轴、阈值线、范围
    for ax in (ax_r, ax_c):
        setup_axis_common(ax)
        ax.set_xlabel("env steps")

    ax_r.set_ylabel("test reward")
    ax_c.set_ylabel("test cost")

    ax_r.set_ylim(*REWARD_YLIM)
    ax_c.set_ylim(*COST_YLIM)

    ax_r.yaxis.set_major_locator(MultipleLocator(REWARD_Y_TICK_STEP))
    ax_c.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))

    ax_c.axhline(COST_THRESHOLD, color="black", linestyle="--", linewidth=1.5, alpha=0.85)

    ax_r.legend(loc="lower right", frameon=True)
    ax_c.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_png = os.path.join(os.getcwd(), "manual_before_after.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"[Saved figure] {out_png}")

if __name__ == "__main__":
    main()
