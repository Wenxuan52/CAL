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
TASKS = ["PointPush1", "PointButton2"]

TASK_MAX_STEP = {
    "PointPush1":   200_000,
    "PointButton2": 200_000,
}

TASK_X_TICK_STEP = {
    "PointPush1":   50_000,
    "PointButton2": 50_000,
}

# 固定 y 轴范围（不想固定就设 None）
TASK_REWARD_YLIM = {
    "PointPush1":   (-11, 12),
    "PointButton2": (-11, 41),
}
TASK_COST_YLIM = {
    "PointPush1":   (-5, 150),
    "PointButton2": (-5, 150),
}

# y tick 密度
COST_Y_TICK_STEP = 30
TASK_REWARD_Y_TICK_STEP = {
    "PointPush1":   5,
    "PointButton2": 10,
}

# SafetyGym cost 阈值线
COST_THRESHOLD = 10

# ======================= 消融配置：MC 采样数与颜色 =======================
MC_LIST = [2, 4, 6, 8, 16]
MODEL_LABELS = [f"N={n}" for n in MC_LIST]

MODEL_COLORS = [
    "#5ad7c3",
    "#8a6bc7",
    "#5da7df",
    "#7fd54c",
    "#f2b84b",
]
COLOR_MAP = dict(zip(MODEL_LABELS, MODEL_COLORS))

# band & line
BAND_ALPHA = 0.22
LINE_WIDTH = 1.3
LINE_ALPHA = 0.75

# ======================= 平滑参数（参考 lambda_compare.py） =======================
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

# ======================= 结果路径：每个 task、每个 MC n 对应多个 csv 路径 =======================
ABLATION_HISTORY_PATHS = {
    "PointPush1": {
        "N=2":  [
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC6/2025-12-24_13-08_seed2952/history.csv",
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC6/2025-12-24_13-25_seed1923/history.csv",
        ],
        "N=4":  [
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC4/2025-12-26_17-01_seed3598/history.csv",
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC4/2025-12-26_16-50_seed9120/history.csv",
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC4/2025-12-26_16-18_seed2204/history.csv",
        ],
        "N=6":  [
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC2/2025-12-24_12-56_seed6394/history.csv",
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC6/2025-12-24_12-23_seed2848/history.csv",
        ],
        "N=8":  [
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC8/2025-12-26_16-34_seed5748/history_processed_manual.csv",
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC8/2025-12-26_17-44_seed6777/history.csv",
        ],
        "N=16": [
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC16/2025-12-26_16-17_seed5767/history_processed_manual.csv",
            "../results/Safexp-PointPush1-v0/pointpush1_algd_ablationMC2/2025-12-24_12-56_seed6394/history.csv",
        ],
    },
    "PointButton2": {
        "N=2":  [
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC2/2025-12-24_13-40_seed1910/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC4/2025-12-26_16-37_seed870/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC2/2025-12-24_14-31_seed4723/history.csv",
        ],
        "N=4":  [
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC4/2025-12-26_15-54_seed9501/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC4/2025-12-26_15-53_seed7142/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC2/2025-12-24_13-58_seed3505/history.csv",
        ],
        "N=6":  [
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC6/2025-12-24_13-19_seed2469/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC6/2025-12-24_13-21_seed6096/history.csv",
        ],
        "N=8":  [
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC8/2025-12-26_15-21_seed9431/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC8/2025-12-26_16-51_seed1698/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC8/2025-12-26_17-07_seed5011/history.csv",
        ],
        "N=16": [
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC16/2025-12-26_16-52_seed3070/history.csv",
            "../results/Safexp-PointButton2-v0/pointbutton2_algd_ablationMC16/2025-12-26_17-09_seed3641/history.csv",
        ],
    },
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
    """
    只需要 step/return/cost 三列（训练 cost 的 boxplot 将使用 cost_mean，而不是 training_cost）
    """
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
    """
    统一 steps_grid，然后 np.interp，区间外变 NaN（和你 lambda_compare 一致）
    """
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
    """
    避免 ‘Mean of empty slice’ warning：对“全 NaN 的列”直接给 NaN。
    """
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

        # 保持“没有任何 run 覆盖”的点为 NaN（让曲线自然断开）
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

    ax.fill_between(steps, lower, upper, color=color, alpha=0.22, linewidth=0)
    line, = ax.plot(steps, mean, color=color, linewidth=1.3, alpha=0.75, label=label)
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
    """
    ✅ 用 cost_mean（聚合后的 mean 曲线）来生成 boxplot 数据
    """
    step = agg["step"]
    vals = agg["cost_mean"]
    m = (step >= lo) & (step <= hi) & np.isfinite(vals)
    v = vals[m]
    if v.size == 0:
        return np.array([np.nan])
    return v

# ======================= 主绘图：2 行 3 列（行=task，列=指标） =======================
def plot_ablation_mc(save_path="ablation_mc_safetygym.pdf"):
    row_labels = TASKS
    col_specs = [
        ("test reward", "reward"),
        ("test cost", "cost"),
        ("training cost", "box"),  # boxplot
    ]

    n_rows, n_cols = len(row_labels), len(col_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.0, 6.6), sharex=False)
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    legend_handles = {}

    # boxplot 用 4 段蓝色渐变（沿用 baseline）
    seg_colors = [cm.Blues(x) for x in np.linspace(0.15, 0.85, 4)]

    # 更大的“行标签”字体：用 y-axis label 作为 row label
    ROW_LABEL_FONTSIZE = 22

    for r, task in enumerate(row_labels):
        max_step = TASK_MAX_STEP[task]
        tick_step = TASK_X_TICK_STEP[task]

        ax_r = axes[r, 0]  # reward
        ax_c = axes[r, 1]  # cost
        ax_b = axes[r, 2]  # training cost boxplot

        # 行标签（更大字号）
        ax_r.set_ylabel(task, fontsize=ROW_LABEL_FONTSIZE, rotation=90, labelpad=18)

        # 列标题只放第一行
        if r == 0:
            axes[r, 0].set_title("test reward")
            axes[r, 1].set_title("test cost")
            axes[r, 2].set_title("training cost")

        # ✅ cache：后面 boxplot 要用到每个 label 的 cost_mean
        agg_cache = {}

        # ====== reward / cost 曲线 ======
        for label in MODEL_LABELS:
            paths = ABLATION_HISTORY_PATHS.get(task, {}).get(label, [])
            paths = [p for p in paths if safe_exists(p)]
            if len(paths) == 0:
                continue

            agg = aggregate_runs(paths, max_step=max_step, metrics=("return", "cost"))
            if agg is None:
                continue
            agg_cache[label] = agg

            steps = agg["step"]
            color = COLOR_MAP[label]

            # reward
            h1 = plot_mean_std(
                ax_r, steps,
                agg["return_mean"], agg["return_std"],
                color=color,
                label=label if label not in legend_handles else None,
                clip_lower_to_zero=False
            )

            # cost：只 clip 下界（mean-std）到 0
            plot_mean_std(
                ax_c, steps,
                agg["cost_mean"], agg["cost_std"],
                color=color,
                label=None,
                clip_lower_to_zero=True
            )

            if label not in legend_handles:
                legend_handles[label] = h1

        # ====== 轴设置：reward / cost ======
        for ax in (ax_r, ax_c):
            setup_axis_common(ax, tick_step)
            ax.set_xlim(0, max_step)
            ax.set_xlabel("")

        if COST_THRESHOLD is not None:
            ax_c.axhline(COST_THRESHOLD, color="black", linestyle="--", linewidth=1.5, alpha=0.85)

        # y range
        r_ylim = TASK_REWARD_YLIM.get(task, None)
        if r_ylim is not None:
            ax_r.set_ylim(*r_ylim)

        c_ylim = TASK_COST_YLIM.get(task, None)
        if c_ylim is not None:
            ax_c.set_ylim(*c_ylim)

        # y tick 密度
        ax_c.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))
        ax_b.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))

        r_tick = TASK_REWARD_Y_TICK_STEP.get(task, None)
        if r_tick is not None:
            ax_r.yaxis.set_major_locator(MultipleLocator(r_tick))

        # 去掉 cost/box 的 y label（避免和 row label 打架）
        ax_c.set_ylabel("")
        ax_b.set_ylabel("")

        # ====== training cost boxplot：✅ 用 cost_mean 分段 ======
        ax_b.grid(True, axis="y", alpha=0.3)
        ax_b.set_xlabel("")

        present_labels = [lb for lb in MODEL_LABELS if lb in agg_cache]
        if len(present_labels) == 0:
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            continue

        edges, seg_leg_labels = segment_labels(max_step, n_segments=4)

        x_base = np.arange(len(present_labels), dtype=float)
        offsets = np.linspace(-0.27, 0.27, 4)
        box_width = 0.16

        seg_handles = []
        for s in range(4):
            lo, hi = edges[s], edges[s + 1]
            data_for_seg = []
            for label in present_labels:
                v = collect_cost_mean_for_segment(agg_cache[label], lo, hi)
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

        ax_b.set_xlim(-0.6, len(present_labels) - 0.4)
        ax_b.set_xticks(x_base)
        ax_b.set_xticklabels(present_labels, rotation=0, ha="center")

        if COST_THRESHOLD is not None:
            ax_b.axhline(COST_THRESHOLD, color="black", linestyle="--", linewidth=1.2, alpha=0.85)

        ax_b.legend(
            seg_handles,
            seg_leg_labels,
            loc="upper right",
            frameon=True,
            fontsize=9,
        )

    # ====== 顶部统一 legend（按 MC_LIST 顺序） ======
    ordered_handles = []
    ordered_labels = []
    for lb in MODEL_LABELS:
        if lb in legend_handles:
            ordered_handles.append(legend_handles[lb])
            ordered_labels.append(lb)

    leg = fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=len(MODEL_LABELS),
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.06),
        borderpad=0.6,
        handlelength=2.2,
        columnspacing=1.4,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("grey")
    frame.set_linewidth(1.5)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.22, hspace=0.25)

    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")

if __name__ == "__main__":
    plot_ablation_mc("ablation_mc_safetygym.pdf")
