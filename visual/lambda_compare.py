import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ======================= 全局风格 =======================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 11,
})

def k_formatter(x, pos):
    x = float(x)
    if abs(x) >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))

# ======================= 路径 =======================
POINTBUTTON1_AUGLAG_PATHS = [
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO4.0/2025-12-31_11-55_seed7682/history.csv",
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO2.0/2025-12-31_11-36_seed999/history.csv",
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.5/2025-12-31_11-59_seed5237/history.csv",
]
POINTBUTTON1_LAG_PATHS = [
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_lambda/2025-12-20_15-47_seed9702/history.csv",
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_lambda/2025-12-20_15-48_seed1932/history.csv",
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_lambda/2025-12-20_15-56_seed7809/history.csv",
    "../results/Safexp-PointButton1-v0/pointbutton1_algd_lambda/2025-12-20_15-59_seed8822/history.csv",
]

HOPPER_AUGLAG_PATHS = [
    "../results/Hopper-v3/hopper_algd_augmentedlambda/2025-12-23_12-16_seed8003/history.csv",
    "../results/Hopper-v3/hopper_algd_augmentedlambda/2025-12-23_11-31_seed8196/history.csv",
    "../results/Hopper-v3/hopper_algd/2025-12-08_03-44_seed8411/history.csv",
    "../results/Hopper-v3/hopper_algd/2025-12-08_03-21_seed1538/history.csv",
]
HOPPER_LAG_PATHS = [
    "../results/Hopper-v3/hopper_algd_lambda/2025-12-23_11-57_seed8797/history.csv",
    "../results/Hopper-v3/hopper_algd_lambda/2025-12-23_12-01_seed3489/history.csv",
    "../results/Hopper-v3/hopper_algd_lambda/2025-12-23_12-48_seed5067/history.csv",
]

# ======================= 颜色（蓝/橙） =======================
COLOR_LAG = "#f0bf4a"
COLOR_AUG = "#468ec9"

# band & line 风格
BAND_ALPHA = 0.25
LINE_WIDTH = 1.5
LINE_ALPHA = 0.75

YLABEL_FONTSIZE = 14
DEFAULT_X_TICK_STEP = 25_000

# ======================= 预算阈值（用于从 cost 兜底计算 violation_mean） =======================
COST_LIMIT = {
    "PointButton1": 10.0,
    "Hopper": 82.748,
}

# ======================= 固定 y 轴范围 =======================
# 现在是 3 列：reward / lambda / violation_mean
TASK_YLIM = {
    "PointButton1": {
        "return": (-3, 42),
        "lambda": (-0.02, 0.14),
        "violation_mean": (-0.05, 2.0),
    },
    "Hopper": {
        "return": (0, 2000),
        "lambda": (-0.02, 1.0),
        "violation_mean": (-5.0, 40.0),  # Hopper 违反幅度可能更大，你可按图再收紧
    }
}

# ======================= 平滑参数 =======================
MEAN_SMOOTH = 0.65
STD_SMOOTH  = 0.85

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

# ======================= 数据读取与聚合 =======================
def safe_exists(path: str) -> bool:
    return path is not None and str(path).strip() != "" and os.path.exists(path)

def read_one_history(path: str, task_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["step", "return", "cost"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[{path}] missing column: {c}")

    df = df.sort_values("step").drop_duplicates("step").reset_index(drop=True)

    # ---- 目标：保证 df 里有 violation_mean ----
    # 1) 如果已经记录了 violation_mean（例如训练时保存的），直接用
    if "violation_mean" not in df.columns:
        # 2) 否则用 eval cost 兜底计算：max(0, cost - h)
        h = COST_LIMIT.get(task_name, None)
        if h is None:
            raise ValueError(f"Missing COST_LIMIT for task '{task_name}'. Please set COST_LIMIT[\"{task_name}\"]")

        cost = df["cost"].to_numpy(dtype=float)
        df["violation_mean"] = np.maximum(0.0, cost - float(h))

    return df

def align_and_stack(dfs, metric: str, steps_grid=None):
    if steps_grid is None:
        steps_grid = np.unique(np.concatenate([d["step"].to_numpy() for d in dfs]))
        steps_grid = np.sort(steps_grid)

    stacked = []
    for d in dfs:
        s = d["step"].to_numpy()
        if metric not in d.columns:
            y = np.full_like(steps_grid, np.nan, dtype=float)
        else:
            v = d[metric].to_numpy(dtype=float)
            y = np.interp(steps_grid, s, v, left=np.nan, right=np.nan)
        stacked.append(y)

    stacked = np.stack(stacked, axis=0)
    return steps_grid, stacked

def mean_std_ignore_nan(stacked):
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)

def aggregate_runs(paths, task_name: str, metrics=("return", "lambda", "violation_mean")):
    valid_paths = [p for p in paths if safe_exists(p)]
    if len(valid_paths) == 0:
        return None

    dfs = [read_one_history(p, task_name) for p in valid_paths]
    steps_grid = np.unique(np.concatenate([d["step"].to_numpy() for d in dfs]))
    steps_grid = np.sort(steps_grid)

    out = {"step": steps_grid}
    for m in metrics:
        _, stacked = align_and_stack(dfs, m, steps_grid=steps_grid)
        mean, std = mean_std_ignore_nan(stacked)

        mean_s = smooth_ema(mean, MEAN_SMOOTH)
        std_s  = smooth_ema(std,  STD_SMOOTH)

        out[f"{m}_mean"] = mean_s
        out[f"{m}_std"]  = std_s
    return out

# ======================= 绘图工具 =======================
def setup_axis_common(ax, x_tick_step=DEFAULT_X_TICK_STEP):
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))

def plot_mean_std(ax, steps, mean, std, color, label=None, metric_name=None):
    lower = mean - std
    upper = mean + std

    # 仅对 Average Violation（violation_mean）做下界 clip
    if metric_name == "violation_mean":
        mean = np.maximum(mean, 0.0)
        lower = np.maximum(lower, 0.0)
        # upper 不需要 clip；如果你也想保证非负，可以加：upper = np.maximum(upper, 0.0)

    ax.fill_between(steps, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)
    line, = ax.plot(steps, mean, color=color, linewidth=LINE_WIDTH, label=label, alpha=LINE_ALPHA)
    return line


# ======================= 主绘图：2 行 3 列 =======================
def plot_lambda_compare(save_path="lambda_compare.pdf"):
    tasks = [
        ("PointButton1", POINTBUTTON1_AUGLAG_PATHS, POINTBUTTON1_LAG_PATHS),
        ("Hopper",       HOPPER_AUGLAG_PATHS,       HOPPER_LAG_PATHS),
    ]

    metric_names = ["return", "lambda", "violation_mean"]
    plot_col_titles = ["Reward", "Lambda", "Average Violation"]

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
        sharex=False
    )

    legend_handles = {}

    for row, (task_name, aug_paths, lag_paths) in enumerate(tasks):
        aug = aggregate_runs(aug_paths, task_name=task_name, metrics=tuple(metric_names))
        lag = aggregate_runs(lag_paths, task_name=task_name, metrics=tuple(metric_names))

        if (aug is None) and (lag is None):
            for c in range(3):
                axes[row, c].axis("off")
            continue

        for col, m in enumerate(metric_names):
            ax = axes[row, col]
            setup_axis_common(ax)

            ax.set_title(plot_col_titles[col])

            if col == 0:
                ax.set_ylabel(task_name, fontsize=YLABEL_FONTSIZE)
            else:
                ax.set_ylabel("")

            if task_name in TASK_YLIM and m in TASK_YLIM[task_name]:
                ax.set_ylim(*TASK_YLIM[task_name][m])

            # Lag / AugLag
            if lag is not None:
                h = plot_mean_std(
                    ax, lag["step"], lag[f"{m}_mean"], lag[f"{m}_std"],
                    color=COLOR_LAG,
                    label="Lagrange" if "Lagrange" not in legend_handles else None,
                    metric_name=m
                )
                if "Lagrange" not in legend_handles:
                    legend_handles["Lagrange"] = h

            if aug is not None:
                h = plot_mean_std(
                    ax, aug["step"], aug[f"{m}_mean"], aug[f"{m}_std"],
                    color=COLOR_AUG,
                    label="Augmented Lagrange" if "Augmented Lagrange" not in legend_handles else None,
                    metric_name=m
                )
                if "Augmented Lagrange" not in legend_handles:
                    legend_handles["Augmented Lagrange"] = h

    ordered_labels = ["Augmented Lagrange", "Lagrange"]
    ordered_handles = [legend_handles[k] for k in ordered_labels if k in legend_handles]

    fig.legend(
        ordered_handles,
        ordered_labels[:len(ordered_handles)],
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.02),
        borderpad=0.6,
        handlelength=2.2,
        columnspacing=2.0,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")

if __name__ == "__main__":
    plot_lambda_compare("lambda_compare.pdf")
