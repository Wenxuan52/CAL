import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =======================
# 手动填一组 history.csv 路径
# =======================
INPUT_HISTORY_CSVS = [
    "../results/Hopper-v3/hopper_algd_augmentedlambda/2025-12-23_12-16_seed8003/history.csv",
    # "../results/Hopper-v3/hopper_algd_augmentedlambda/2025-12-23_11-31_seed8196/history.csv",
    # "../results/Hopper-v3/hopper_algd/2025-12-08_03-44_seed8411/history.csv",
    "../results/Hopper-v3/hopper_algd/2025-12-08_03-21_seed1538/history_processed_manual.csv",
]

# 可选：截断到某个 max_step（不需要就设 None）
MAX_STEP = None

# 输出文件名（会输出到“脚本所在文件夹”）
OUT_CSV_NAME = "history_meanstd.csv"
OUT_PNG_NAME = "history_meanstd.png"


# =======================
# 平滑参数：与你最新绘图脚本一致
# =======================
MEAN_SMOOTH = 0.75
STD_SMOOTH  = 0.88

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
})


def _pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def smooth_ema(values, weight=0.8):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v

    out = np.copy(v)
    idx0 = np.where(~np.isnan(v))[0]
    if idx0.size == 0:
        return v

    i0 = idx0[0]
    out[:i0] = np.nan
    out[i0] = v[i0]

    last = v[i0]
    for i in range(i0 + 1, len(v)):
        if np.isnan(v[i]):
            out[i] = np.nan
            continue
        last = weight * last + (1.0 - weight) * v[i]
        out[i] = last
    return out


def mean_std_ignore_nan_safe(stacked: np.ndarray):
    stacked = np.asarray(stacked, dtype=float)
    valid = np.any(np.isfinite(stacked), axis=0)
    mean = np.full(stacked.shape[1], np.nan, dtype=float)
    std  = np.full(stacked.shape[1], np.nan, dtype=float)
    if np.any(valid):
        mean[valid] = np.nanmean(stacked[:, valid], axis=0)
        std[valid]  = np.nanstd(stacked[:, valid], axis=0)
    return mean, std, valid


def read_one_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    step_col = _pick_first_existing_col(df, ["step", "total_step", "timesteps", "env_step", "global_step", "Step"])
    ret_col  = _pick_first_existing_col(df, ["return", "test_reward", "reward", "ep_ret", "return_mean"])
    cost_col = _pick_first_existing_col(df, ["cost", "test_cost", "ep_cost", "cost_mean"])

    if step_col is None or ret_col is None or cost_col is None:
        raise ValueError(f"[{path}] missing required columns. have={list(df.columns)}")

    out = df[[step_col, ret_col, cost_col]].copy()
    out.columns = ["step", "return", "cost"]
    out = out.sort_values("step").drop_duplicates("step").reset_index(drop=True)
    return out


def align_and_stack(dfs: List[pd.DataFrame], metric: str, steps_grid: np.ndarray):
    stacked = []
    for d in dfs:
        s = d["step"].to_numpy(dtype=float)
        v = d[metric].to_numpy(dtype=float)
        y = np.interp(steps_grid, s, v, left=np.nan, right=np.nan)
        stacked.append(y)
    return np.stack(stacked, axis=0)


def aggregate_histories(paths: List[str], max_step: Optional[int] = None) -> pd.DataFrame:
    dfs = [read_one_history(p) for p in paths]

    if max_step is not None:
        for d in dfs:
            d.drop(d[d["step"] > max_step].index, inplace=True)

    steps_grid = np.unique(np.concatenate([d["step"].to_numpy() for d in dfs]))
    steps_grid = np.sort(steps_grid)

    # return
    ret_stacked = align_and_stack(dfs, "return", steps_grid)
    ret_mean, ret_std, ret_valid = mean_std_ignore_nan_safe(ret_stacked)
    ret_mean_s = smooth_ema(ret_mean, MEAN_SMOOTH)
    ret_std_s  = smooth_ema(ret_std,  STD_SMOOTH)
    ret_mean_s[~ret_valid] = np.nan
    ret_std_s[~ret_valid]  = np.nan

    # cost
    cost_stacked = align_and_stack(dfs, "cost", steps_grid)
    cost_mean, cost_std, cost_valid = mean_std_ignore_nan_safe(cost_stacked)
    cost_mean_s = smooth_ema(cost_mean, MEAN_SMOOTH)
    cost_std_s  = smooth_ema(cost_std,  STD_SMOOTH)
    cost_mean_s[~cost_valid] = np.nan
    cost_std_s[~cost_valid]  = np.nan

    # lower/upper：匹配你的 load_history()
    ret_lower  = ret_mean_s - ret_std_s
    ret_upper  = ret_mean_s + ret_std_s

    cost_lower = cost_mean_s - cost_std_s
    cost_upper = cost_mean_s + cost_std_s

    # 和你主图一致：cost band 下界 clip 到 0
    cost_lower = np.maximum(cost_lower, 0.0)

    out = pd.DataFrame({
        "step": steps_grid.astype(int),
        "return_mean": ret_mean_s,
        "return_std_lower": ret_lower,
        "return_std_upper": ret_upper,
        "cost_mean": cost_mean_s,
        "cost_std_lower": cost_lower,
        "cost_std_upper": cost_upper,
    })
    return out


def plot_meanstd(df: pd.DataFrame, save_path: str):
    step = df["step"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # return
    ax = axes[0]
    m = df["return_mean"].to_numpy()
    lo = df["return_std_lower"].to_numpy()
    up = df["return_std_upper"].to_numpy()
    ax.fill_between(step, lo, up, alpha=0.22, linewidth=0)
    ax.plot(step, m, linewidth=1.6, alpha=0.85)
    ax.set_title("Return (mean ± std)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    # cost
    ax = axes[1]
    m = df["cost_mean"].to_numpy()
    lo = df["cost_std_lower"].to_numpy()
    up = df["cost_std_upper"].to_numpy()
    ax.fill_between(step, lo, up, alpha=0.22, linewidth=0)
    ax.plot(step, m, linewidth=1.6, alpha=0.85)
    ax.set_title("Cost (mean ± std)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")


def main():
    if len(INPUT_HISTORY_CSVS) == 0:
        raise ValueError("请在脚本顶部 INPUT_HISTORY_CSVS 里填入至少一个 history.csv 路径。")

    paths = [str(p) for p in INPUT_HISTORY_CSVS]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # ✅ 输出到脚本所在文件夹
    script_dir = Path(__file__).resolve().parent
    out_csv = script_dir / OUT_CSV_NAME
    out_png = script_dir / OUT_PNG_NAME

    df_out = aggregate_histories(paths, max_step=MAX_STEP)
    df_out.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    plot_meanstd(df_out, str(out_png))


if __name__ == "__main__":
    main()
