import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.cm as cm

# ======================= 全局风格 =======================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    # "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 18,
})

COST_Y_TICK_STEP = 40
TASK_REWARD_Y_TICK_STEP = {
    "PointButton1": 10,
    "CarButton1":   10,
    "PointButton2": 10,
    "PointPush1":   5,
    "CarButton2":   10,
}

# ======================= Baselines 配置 =======================
MODEL_LABELS = [
    "PPO + Lag",
    "ALGD (Ours)",
]

# 颜色与 label 一一对应
MODEL_COLORS = [
    # "#5ad7c3",
    # "#8a6bc7",  
    "#5da7df",  
    # "#7fd54c",
    "#d64a4b",
]
COLOR_MAP = dict(zip(MODEL_LABELS, MODEL_COLORS))

# “ours” 高亮，其余半透明（你也可改）
ALPHA_HIGHLIGHT = 1.0
ALPHA_OTHERS = 0.55
BAND_ALPHA = 0.15
OURS_BAND_ALPHA = 0.22

# Cost 阈值线
COST_THRESHOLD = 10

# ======================= Task 配置 =======================
# 你要画的列顺序
TASKS = [
    "PointButton1",
    "CarButton1",
    "PointButton2",
    "PointPush1",
    "CarButton2",
]

# 每个 task 的最大步数（用于 xlim & 分段）
TASK_MAX_STEP = {
    "PointButton1": 100_000,
    "CarButton1": 150_000,
    "PointButton2": 200_000,
    "PointPush1":   200_000,
    "CarButton2":   300_000,
}

# 每个 task 的 x tick 步长（更接近你图里的效果）
TASK_X_TICK_STEP = {
    "PointButton1": 25_000,
    "CarButton1":   50_000,
    "PointButton2": 50_000,
    "PointPush1":   50_000,
    "CarButton2":   100_000,
}

# ======================= 新增：每个 task 的 y 轴范围接口 =======================
TASK_REWARD_YLIM = {
    "PointButton1": (-21, 42),
    "CarButton1":   (-6, 32),
    "PointButton2": (-10, 45),
    "PointPush1":   (-12, 8),
    "CarButton2":   (-6, 35),
}

TASK_COST_YLIM = {
    "PointButton1": (-5, 100),        # 例如 (0, 100)
    "CarButton1":   (-5, 120),        # 例如 (0, 120)
    "PointButton2": (-5, 210),        # 例如 (0, 120)
    "PointPush1":   (-5, 210),        # 例如 (0, 120)
    "CarButton2":   (-5, 210),        # 例如 (0, 160)
}


# ======================= history路径 =======================
TASK_HISTORY_PATHS = {
    "PointButton1": {
        "PPO + Lag":     "../results/Safexp-PointButton1-v0/pointbutton1_ppolag/2025-12-17_20-13_seed9224/history_meanstd.csv",
        "ALGD (Ours)":   "../results/Safexp-PointButton1-v0/pointbutton1_algd/2025-12-04_23-21_seed2861/history_meanstd.csv",
    },
    "CarButton1": {
        "PPO + Lag":     "../results/Safexp-CarButton1-v0/carbutton1_ppolag/2025-12-17_20-27_seed2303/history_meanstd.csv",
        "ALGD (Ours)":   "../results/Safexp-CarButton1-v0/carbutton1_algd_MC/2025-11-28_23-04_seed9494/history_meanstd.csv",
    },
    "PointButton2": {
        "PPO + Lag":     "../results/Safexp-PointButton2-v0/pointbutton2_ppolag/2025-12-17_20-22_seed8657/history_meanstd.csv",
        "ALGD (Ours)":   "temp_history/pointbutton2_history_meanstd.csv",
    },
    "PointPush1": {
        "PPO + Lag":     "../results/Safexp-PointPush1-v0/pointpush1_ppolag/2025-12-17_20-26_seed9300/history_meanstd.csv",
        "ALGD (Ours)":   "temp_history/pointpush1_history_meanstd.csv",
    },
    "CarButton2": {
        "PPO + Lag":     "../results/Safexp-CarButton2-v0/carbutton2_ppolag/2025-12-17_20-41_seed548/history_meanstd.csv",
        "ALGD (Ours)":   "../results/Safexp-CarButton2-v0/carbutton2_algd/2025-12-09_12-07_seed7999/history_meanstd.csv",
    },
}

# ======================= 工具函数 =======================
def k_formatter(x, pos):
    x = float(x)
    if abs(x) >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))

# 这些平滑/伪 band 函数可以保留（不再使用也不影响）
def smooth_ema(values, weight=0.8):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or weight <= 0:
        return values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed

def compute_pseudo_band_single_run(values, mean_smoothing=0.8):
    values = np.asarray(values, dtype=float)
    T = len(values)
    if T == 0:
        return values, values, values

    mean = smooth_ema(values, mean_smoothing)
    diff = values - mean
    idx = np.arange(T)

    pos_mask = diff >= 0
    neg_mask = diff < 0

    def interp_from_mask(mask):
        if not np.any(mask):
            return np.zeros(T, dtype=float)
        x = idx[mask]
        y = diff[mask]
        if x.size == 1:
            return np.full(T, y[0], dtype=float)
        return np.interp(idx, x, y)

    upper_offset = interp_from_mask(pos_mask)
    lower_offset = interp_from_mask(neg_mask)

    upper = mean + upper_offset
    lower = mean + lower_offset
    return mean, lower, upper

# ======================= ✅ 更新：直接读取 mean/std_upper/std_lower =======================
def load_history(path):
    df = pd.read_csv(path)

    step = df["step"].to_numpy()

    # return
    ret_mean  = df["return_mean"].to_numpy()
    ret_upper = df["return_std_upper"].to_numpy()
    ret_lower = df["return_std_lower"].to_numpy()

    # cost
    cost_mean  = df["cost_mean"].to_numpy()
    cost_upper = df["cost_std_upper"].to_numpy()
    cost_lower = df["cost_std_lower"].to_numpy()

    return step, ret_mean, ret_lower, ret_upper, cost_mean, cost_lower, cost_upper

def safe_exists(path):
    return path is not None and str(path).strip() != "" and os.path.exists(path)

def clip_by_max_step(step, arr, max_step):
    mask = step <= max_step
    return step[mask], arr[mask]

def segment_labels(max_step, n_segments=4):
    edges = np.linspace(0, max_step, n_segments + 1, dtype=int)
    labels = []
    for i in range(n_segments):
        a = edges[i]
        b = edges[i + 1]
        labels.append(f"{int(a/1000)}-{int(b/1000)}k")
    return edges, labels

# ======================= 主绘图 =======================
def main(save_path="all_tasks_compare.pdf"):
    n_rows, n_cols = 3, len(TASKS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 9), sharex=False)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    formatter = FuncFormatter(k_formatter)

    # 顶部统一 legend：只收集一次（按 MODEL_LABELS 顺序）
    legend_handles = {}

    # boxplot 用 4 段蓝色渐变
    seg_colors = [cm.Blues(x) for x in np.linspace(0.15, 0.85, 4)]

    for col, task in enumerate(TASKS):
        max_step = TASK_MAX_STEP[task]
        tick_step = TASK_X_TICK_STEP[task]

        ax_r = axes[0, col]
        ax_c = axes[1, col]
        ax_b = axes[2, col]

        ax_r.set_title(task)

        # ====== Reward / Cost 曲线 ======
        for label in MODEL_LABELS:
            path = TASK_HISTORY_PATHS.get(task, {}).get(label, None)
            if not safe_exists(path):
                continue

            # ✅ 直接读出 mean/lower/upper
            step, ret_mean, ret_lower, ret_upper, cost_mean, cost_lower, cost_upper = load_history(path)

            # ✅ 与原逻辑一致：按 max_step 裁剪
            mask = step <= max_step
            step_r = step[mask]
            step_c = step[mask]

            ret_mean  = ret_mean[mask]
            ret_lower = ret_lower[mask]
            ret_upper = ret_upper[mask]

            cost_mean  = cost_mean[mask]
            cost_lower = cost_lower[mask]
            cost_upper = cost_upper[mask]

            color = COLOR_MAP[label]
            is_ours = (label == "ALGD (Ours)")
            alpha_line = ALPHA_HIGHLIGHT if is_ours else ALPHA_OTHERS

            ax_r.fill_between(step_r, ret_lower, ret_upper, color=color, alpha=OURS_BAND_ALPHA if is_ours else BAND_ALPHA, linewidth=0)
            line_r, = ax_r.plot(step_r, ret_mean, color=color, linewidth=2, alpha=alpha_line)

            ax_c.fill_between(step_c, cost_lower, cost_upper, color=color, alpha=OURS_BAND_ALPHA if is_ours else BAND_ALPHA, linewidth=0)
            line_c, = ax_c.plot(step_c, cost_mean, color=color, linewidth=2, alpha=alpha_line)

            # legend 句柄全局只收集一次
            if label not in legend_handles:
                legend_handles[label] = line_c

        # 网格、阈值线
        ax_r.grid(True, alpha=0.3)
        ax_c.grid(True, alpha=0.3)
        if COST_THRESHOLD is not None:
            ax_c.axhline(COST_THRESHOLD, color="black", linestyle="--", linewidth=1.5, alpha=0.85)

        # x 轴：范围、刻度、k 格式；不显示 xlabel
        for ax in (ax_r, ax_c):
            ax.set_xlim(0, max_step)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(MultipleLocator(tick_step))
            ax.set_xlabel("")

        # ====== 新增：按 task 设置 y 轴范围 ======
        r_ylim = TASK_REWARD_YLIM.get(task, None)
        if r_ylim is not None:
            ax_r.set_ylim(r_ylim[0], r_ylim[1])

        c_ylim = TASK_COST_YLIM.get(task, None)
        if c_ylim is not None:
            ax_c.set_ylim(c_ylim[0], c_ylim[1])
            
        # ====== 让 y 轴多一些刻度 ======
        ax_c.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))  # test cost 每 20
        ax_b.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))  # training cost (boxplot) 也每 20（可选但建议一致）

        r_tick = TASK_REWARD_Y_TICK_STEP.get(task, None)
        if r_tick is not None:
            ax_r.yaxis.set_major_locator(MultipleLocator(r_tick))        # reward 按 task 设置 10/5/...

        # y 标签只放第一列
        if col == 0:
            ax_r.set_ylabel("test reward")
            ax_c.set_ylabel("test cost")
            ax_b.set_ylabel("training cost")
        else:
            ax_r.set_ylabel("")
            ax_c.set_ylabel("")
            ax_b.set_ylabel("")

        # ====== Boxplot：把总 step 均分 4 段 ======
        ax_b.grid(True, axis="y", alpha=0.3)
        ax_b.set_xlabel("")

        present_labels = []
        for label in MODEL_LABELS:
            path = TASK_HISTORY_PATHS.get(task, {}).get(label, None)
            if safe_exists(path):
                present_labels.append(label)

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
            data_for_seg = []
            for label in present_labels:
                # ✅ 直接读取 cost_mean（用于段内 boxplot）
                step, _, _, _, cost_mean, _, _ = load_history(TASK_HISTORY_PATHS[task][label])

                lo, hi = edges[s], edges[s + 1]
                mask = (step >= lo) & (step <= hi)
                vals = cost_mean[mask]
                if vals.size == 0:
                    vals = np.array([np.nan])
                data_for_seg.append(vals)

            positions = x_base + offsets[s]
            bp = ax_b.boxplot(
                data_for_seg,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                flierprops=dict(
                    marker='D',                 # 菱形
                    markersize=4,               # 大小可调
                    markerfacecolor="black",  # 实心填充（也可以用 "black"）
                    markeredgecolor="black",  # 边框同色
                    markeredgewidth=0.0,        # 不要空心效果
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
        ax_b.set_xticklabels(present_labels, rotation=20, ha="right")

        if COST_THRESHOLD is not None:
            ax_b.axhline(COST_THRESHOLD, color="black", linestyle="--", linewidth=1.2, alpha=0.85)

        ax_b.legend(
            seg_handles,
            seg_leg_labels,
            loc="upper right",
            frameon=True,
            fontsize=9,
        )

    # ====== 顶部统一 legend（新增：黑色圆角矩形框） ======
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
        frameon=True,          # 开启边框
        fancybox=True,         # 圆角
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
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main("onpolicy_safetygym.pdf")
