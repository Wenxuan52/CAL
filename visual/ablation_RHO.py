import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.cm as cm

# ======================= 全局风格（沿用 baseline 大图） =======================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 12,
    "legend.fontsize": 18,
})

def k_formatter(x, pos):
    x = float(x)
    if abs(x) >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))

# ======================= 配置：任务与步数 =======================
TASKS = ["PointButton1", "Hopper"]

TASK_MAX_STEP = {
    "PointButton1": 100_000,
    "Hopper": 150_000,
}

TASK_X_TICK_STEP = {
    "PointButton1": 20_000,
    "Hopper": 50_000,
}

# 固定 y 轴范围（不想固定就设 None）
TASK_REWARD_YLIM = {
    "PointButton1": (-11, 41),
    "Hopper": (-50, 2100),
}
TASK_COST_YLIM = {
    "PointButton1": (-5, 60),
    "Hopper": (-5, 100),
}

# ======================= 每个 task 的 cost y-tick step（你要的：PB1=10, Hopper=40） =======================
TASK_COST_Y_TICK_STEP = {
    "PointButton1": 10,
    "Hopper": 40,
}

# reward y tick（如不固定就 None）
TASK_REWARD_Y_TICK_STEP = {
    "PointButton1": None,
    "Hopper": None,
}

# ======================= 每个 task 的 cost 阈值线（你要的：PB1=10, Hopper=82.748） =======================
TASK_COST_THRESHOLD = {
    "PointButton1": 10,
    "Hopper": 82.748,
}

# ======================= 消融配置：rho 与颜色 =======================
RHO_LIST = [0.1, 0.5, 1.0, 2.0, 4.0]

MODEL_KEYS = [f"rho={r}" for r in RHO_LIST]
MODEL_LABELS = [rf"$\rho={r}$" for r in RHO_LIST]

MODEL_COLORS = [
    "#5ad7c3",
    "#8a6bc7",
    "#5da7df",
    "#7fd54c",
    "#f2b84b",
]

COLOR_MAP = dict(zip(MODEL_KEYS, MODEL_COLORS))
LABEL_MAP = dict(zip(MODEL_KEYS, MODEL_LABELS))

BAND_ALPHA = 0.22
LINE_WIDTH = 1.3
LINE_ALPHA = 0.75

# ======================= 平滑参数 =======================
MEAN_SMOOTH = 0.75
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

# ======================= 结果路径：每个 rho 对应多个 csv 路径（手动填） =======================
ABLATION_HISTORY_PATHS = {
    "PointButton1": {
        "rho=0.1": [
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.1/2025-12-31_11-28_seed5000/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.1/2025-12-31_11-52_seed4987/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.1/2025-12-31_12-18_seed1386/history.csv",
        ],
        "rho=0.5": [
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.5/2025-12-31_11-28_seed8477/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.5/2025-12-31_11-59_seed5237/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO0.5/2025-12-31_12-26_seed9758/history.csv",
        ],
        "rho=1.0": [
            "../results/Safexp-PointButton1-v0/pointbutton1_algd/2025-12-04_23-07_seed8856/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd/2025-12-04_23-09_seed175/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd/2025-12-04_23-21_seed2861/history.csv",
        ],
        "rho=2.0": [
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO2.0/2025-12-31_11-36_seed999/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO2.0/2025-12-31_11-51_seed3198/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO2.0/2025-12-31_11-58_seed4538/history.csv",
        ],
        "rho=4.0": [
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO4.0/2025-12-31_11-55_seed7682/history.csv",
            "../results/Safexp-PointButton1-v0/pointbutton1_algd_ablationRHO4.0/2025-12-31_12-21_seed1260/history.csv",
        ],
    },
    "Hopper": {
        "rho=0.1": [
            "../results/Hopper-v3/hopper_algd_ablationRHO0.1/2026-01-04_15-11_seed6559/history.csv",
            "../results/Hopper-v3/hopper_algd_ablationRHO0.1/2026-01-04_15-50_seed2532/history.csv",
        ],
        "rho=0.5": [
            "../results/Hopper-v3/hopper_algd_ablationRHO0.5/2026-01-04_14-40_seed2036/history.csv",
            "../results/Hopper-v3/hopper_algd_ablationRHO0.5/2026-01-04_15-26_seed576/history.csv",
            "../results/Hopper-v3/hopper_algd_ablationRHO0.5/2026-01-04_15-32_seed4689/history.csv",
        ],
        "rho=1.0": [
            "../results/Hopper-v3/hopper_algd_augmentedlambda/2025-12-23_12-16_seed8003/history.csv",
            "../results/Hopper-v3/hopper_algd/2025-12-08_03-21_seed1538/history_processed_manual.csv",
        ],
        "rho=2.0": [
            "../results/Hopper-v3/hopper_algd_ablationRHO0.5/2026-01-04_15-32_seed4689/history.csv",
            "../results/Hopper-v3/hopper_algd_ablationRHO2.0/2026-01-04_15-28_seed9971/history.csv",
        ],
        "rho=4.0": [
            "../results/Hopper-v3/hopper_algd_ablationRHO4.0/2026-01-04_15-14_seed1581/history.csv",
            "../results/Hopper-v3/hopper_algd_ablationRHO4.0/2026-01-04_15-38_seed1513/history.csv",
        ],
    }
}

# ======================= 数据读取与聚合（多 csv -> align -> mean/std -> smooth） =======================
def safe_exists(path: str) -> bool:
    return path is not None and str(path).strip() != "" and os.path.exists(path)

def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def read_one_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    step_col = _pick_first_existing_col(df, ["step", "total_step", "timesteps", "env_step", "global_step", "Step"])
    ret_col  = _pick_first_existing_col(df, ["return", "test_reward", "reward", "ep_ret", "return_mean"])
    cost_col = _pick_first_existing_col(df, ["cost", "test_cost", "ep_cost", "cost_mean"])

    if step_col is None or ret_col is None or cost_col is None:
        raise ValueError(f"[{path}] missing required columns. have={list(df.columns)}")

    df = df[[step_col, ret_col, cost_col]].copy()
    df.columns = ["step", "return", "cost"]
    df = df.sort_values("step").drop_duplicates("step").reset_index(drop=True)
    return df

def align_and_stack(dfs, metric: str, steps_grid=None):
    if steps_grid is None:
        steps_grid = np.unique(np.concatenate([d["step"].to_numpy() for d in dfs]))
        steps_grid = np.sort(steps_grid)

    stacked = []
    for d in dfs:
        s = d["step"].to_numpy(dtype=float)
        v = d[metric].to_numpy(dtype=float)
        y = np.interp(steps_grid, s, v, left=np.nan, right=np.nan)
        stacked.append(y)

    stacked = np.stack(stacked, axis=0)
    return steps_grid, stacked

def mean_std_ignore_nan_safe(stacked):
    stacked = np.asarray(stacked, dtype=float)
    valid = np.any(np.isfinite(stacked), axis=0)
    mean = np.full(stacked.shape[1], np.nan, dtype=float)
    std  = np.full(stacked.shape[1], np.nan, dtype=float)
    if np.any(valid):
        mean[valid] = np.nanmean(stacked[:, valid], axis=0)
        std[valid]  = np.nanstd(stacked[:, valid], axis=0)
    return mean, std, valid

def aggregate_runs(paths, max_step: int, metrics=("return", "cost")):
    valid_paths = [p for p in paths if safe_exists(p)]
    if len(valid_paths) == 0:
        return None

    dfs = [read_one_history(p) for p in valid_paths]
    for d in dfs:
        d.drop(d[d["step"] > max_step].index, inplace=True)

    steps_grid = np.unique(np.concatenate([d["step"].to_numpy() for d in dfs]))
    steps_grid = np.sort(steps_grid)

    out = {"step": steps_grid}

    for m in metrics:
        _, stacked = align_and_stack(dfs, m, steps_grid=steps_grid)
        mean, std, valid = mean_std_ignore_nan_safe(stacked)

        mean_s = smooth_ema(mean, MEAN_SMOOTH)
        std_s  = smooth_ema(std,  STD_SMOOTH)

        mean_s[~valid] = np.nan
        std_s[~valid]  = np.nan

        out[f"{m}_mean"] = mean_s
        out[f"{m}_std"]  = std_s

    return out

# ======================= 绘图工具 =======================
def setup_axis_common(ax, x_tick_step):
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))

def plot_mean_std(ax, steps, mean, std, color, label=None, clip_lower_to_zero=False):
    lower = mean - std
    upper = mean + std

    if clip_lower_to_zero:
        lower = np.maximum(lower, 0.0)

    ax.fill_between(steps, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)
    line, = ax.plot(steps, mean, color=color, linewidth=LINE_WIDTH, alpha=LINE_ALPHA, label=label)
    return line

def segment_labels(max_step, n_segments=4):
    edges = np.linspace(0, max_step, n_segments + 1, dtype=int)
    labels = []
    for i in range(n_segments):
        a = edges[i]
        b = edges[i + 1]
        labels.append(f"{int(a/1000)}-{int(b/1000)}k")
    return edges, labels

def collect_cost_mean_for_segment(agg, lo, hi):
    step = agg["step"]
    vals = agg["cost_mean"]
    m = (step >= lo) & (step <= hi) & np.isfinite(vals)
    v = vals[m]
    if v.size == 0:
        return np.array([np.nan])
    return v

# ======================= 主绘图：n 行 3 列 =======================
def plot_ablation_rho(save_path="ablation_rho_safetygym.pdf"):
    row_labels = TASKS
    col_specs = [
        ("test reward", "reward"),
        ("test cost", "cost"),
        ("training cost", "box"),
    ]

    n_rows, n_cols = len(row_labels), len(col_specs)

    # ✅ 图尺寸随行数变化（更贴合两行/多行）
    fig_w = 16.0
    fig_h_per_row = 3.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, 6.6), sharex=False)
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    legend_handles = {}
    seg_colors = [cm.Blues(x) for x in np.linspace(0.15, 0.85, 4)]
    ROW_LABEL_FONTSIZE = 22

    for r, task in enumerate(row_labels):
        max_step = TASK_MAX_STEP[task]
        tick_step = TASK_X_TICK_STEP[task]

        # ✅ task-specific cost tick & threshold
        cost_tick_step = TASK_COST_Y_TICK_STEP.get(task, 10)
        cost_threshold = TASK_COST_THRESHOLD.get(task, None)

        ax_r = axes[r, 0]
        ax_c = axes[r, 1]
        ax_b = axes[r, 2]

        ax_r.set_ylabel(task, fontsize=ROW_LABEL_FONTSIZE, rotation=90, labelpad=18)

        if r == 0:
            axes[r, 0].set_title("test reward")
            axes[r, 1].set_title("test cost")
            axes[r, 2].set_title("training cost")

        agg_cache = {}

        # ====== reward / cost 曲线 ======
        for key in MODEL_KEYS:
            paths = ABLATION_HISTORY_PATHS.get(task, {}).get(key, [])
            paths = [p for p in paths if safe_exists(p)]
            if len(paths) == 0:
                continue

            agg = aggregate_runs(paths, max_step=max_step, metrics=("return", "cost"))
            if agg is None:
                continue
            agg_cache[key] = agg

            steps = agg["step"]
            color = COLOR_MAP[key]
            latex_label = LABEL_MAP[key]

            h1 = plot_mean_std(
                ax_r, steps,
                agg["return_mean"], agg["return_std"],
                color=color,
                label=latex_label if key not in legend_handles else None,
                clip_lower_to_zero=False
            )

            plot_mean_std(
                ax_c, steps,
                agg["cost_mean"], agg["cost_std"],
                color=color,
                label=None,
                clip_lower_to_zero=True
            )

            if key not in legend_handles:
                legend_handles[key] = h1

        # ====== 轴设置：reward / cost ======
        for ax in (ax_r, ax_c):
            setup_axis_common(ax, tick_step)
            ax.set_xlim(0, max_step)
            ax.set_xlabel("")

        # ✅ task-specific threshold line
        if cost_threshold is not None:
            ax_c.axhline(cost_threshold, color="black", linestyle="--", linewidth=1.5, alpha=0.85)

        r_ylim = TASK_REWARD_YLIM.get(task, None)
        if r_ylim is not None:
            ax_r.set_ylim(*r_ylim)

        c_ylim = TASK_COST_YLIM.get(task, None)
        if c_ylim is not None:
            ax_c.set_ylim(*c_ylim)

        # ✅ task-specific cost tick step
        ax_c.yaxis.set_major_locator(MultipleLocator(cost_tick_step))
        ax_b.yaxis.set_major_locator(MultipleLocator(cost_tick_step))

        r_tick = TASK_REWARD_Y_TICK_STEP.get(task, None)
        if r_tick is not None:
            ax_r.yaxis.set_major_locator(MultipleLocator(r_tick))

        ax_c.set_ylabel("")
        ax_b.set_ylabel("")

        # ====== training cost boxplot ======
        ax_b.grid(True, axis="y", alpha=0.3)
        ax_b.set_xlabel("")

        present_keys = [k for k in MODEL_KEYS if k in agg_cache]
        if len(present_keys) == 0:
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            continue

        edges, seg_leg_labels = segment_labels(max_step, n_segments=4)

        x_base = np.arange(len(present_keys), dtype=float)
        offsets = np.linspace(-0.27, 0.27, 4)
        box_width = 0.16

        seg_handles = []
        for s in range(4):
            lo, hi = edges[s], edges[s + 1]
            data_for_seg = []
            for key in present_keys:
                v = collect_cost_mean_for_segment(agg_cache[key], lo, hi)
                data_for_seg.append(v)

            positions = x_base + offsets[s]
            bp = ax_b.boxplot(
                data_for_seg,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                flierprops=dict(
                    marker='D',
                    markersize=4,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markeredgewidth=0.0,
                    alpha=0.6
                )
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(seg_colors[s])
                patch.set_edgecolor("black")
                patch.set_alpha(0.95)
            for element in ["whiskers", "caps", "medians"]:
                for item in bp[element]:
                    item.set_color("black")
                    item.set_linewidth(1.0)

            seg_handles.append(bp["boxes"][0])

        ax_b.set_xlim(-0.6, len(present_keys) - 0.4)
        ax_b.set_xticks(x_base)
        ax_b.set_xticklabels([LABEL_MAP[k] for k in present_keys], rotation=0, ha="center")

        # ✅ task-specific threshold line on box plot too
        if cost_threshold is not None:
            ax_b.axhline(cost_threshold, color="black", linestyle="--", linewidth=1.2, alpha=0.85)

        ax_b.legend(
            seg_handles,
            seg_leg_labels,
            loc="best",
            frameon=True,
            fontsize=9,
        )

    # ====== 顶部统一 legend（按 rho 顺序） ======
    ordered_handles = []
    ordered_labels = []
    for k in MODEL_KEYS:
        if k in legend_handles:
            ordered_handles.append(legend_handles[k])
            ordered_labels.append(LABEL_MAP[k])

    # ✅ 两行图时 legend 往上一点，避免遮挡
    leg = fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=len(MODEL_KEYS),
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.02),
        borderpad=0.6,
        handlelength=2.2,
        columnspacing=1.4,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(1.5)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)

    # ✅ 给顶部 legend 留空间（多行时更稳）
    top_margin = 0.90 if n_rows == 1 else 0.93
    plt.tight_layout(rect=[0, 0, 1, top_margin])
    plt.subplots_adjust(wspace=0.22)

    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")

if __name__ == "__main__":
    plot_ablation_rho("ablation_rho_safetygym.pdf")
