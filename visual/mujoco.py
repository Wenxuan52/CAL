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
    "axes.labelsize": 12,
    "legend.fontsize": 18,
})

COST_Y_TICK_STEP = 40

# Mujoco 的 reward tick 你后续可按需要改（先给一个保守默认）
TASK_REWARD_Y_TICK_STEP = {
    "HalfCheetah": 1000,
    "Hopper":      500,
    "Ant":         500,
    "Humanoid":    1000,
}

# ======================= Baselines 配置 =======================
MODEL_LABELS = [
    "SAC + Lag",
    "SAC + AugLag",
    "HJ",
    "CAL",
    "ALGD (Ours)",
]

MODEL_COLORS = [
    "#5ad7c3",
    "#8a6bc7",
    "#5da7df",
    "#7fd54c",
    "#d64a4b",
]
COLOR_MAP = dict(zip(MODEL_LABELS, MODEL_COLORS))

ALPHA_HIGHLIGHT = 1.0
ALPHA_OTHERS = 0.4
BAND_ALPHA = 0.10
OURS_BAND_ALPHA = 0.22

# Cost 阈值线：每个 task 一个
COST_THRESHOLDS = {
    "Ant-v3": 103.115,
    "HalfCheetah-v3": 151.989,
    "Hopper-v3": 82.748,
    "Humanoid-v3": 20.140,
}


# ======================= Mujoco Task 配置 =======================
TASKS = [
    "HalfCheetah",
    "Hopper",
    "Ant",
    "Humanoid",
]

# 这里给常见训练步数占位，你可按你的日志实际步数改
TASK_MAX_STEP = {
    "HalfCheetah": 100_000,
    "Hopper":      150_000,
    "Ant":         200_000,
    "Humanoid":    500_000,
}

TASK_X_TICK_STEP = {
    "HalfCheetah": 25_000,
    "Hopper":      50_000,
    "Ant":         50_000,
    "Humanoid":    100_000,
}

# ======================= y 轴范围接口（先给 None / 占位） =======================
# 如果你希望像 safety-gym 一样每个 task 固定 y 轴范围，就在这里填 (low, high)
TASK_REWARD_YLIM = {
    "HalfCheetah": (-800, 2500),
    "Hopper":      (-500, 3000),
    "Ant":         (-750, 1400),
    "Humanoid":    (-200, 3700),
}

TASK_COST_YLIM = {
    "HalfCheetah": (-5, 250),
    "Hopper":      (-5, 220),
    "Ant":         (-5, 200),
    "Humanoid":    (-3, 37),
}

# ======================= history路径（先留空，后续你填） =======================
TASK_HISTORY_PATHS = {
    "HalfCheetah": {
        "SAC + Lag":     "../results/HalfCheetah-v3/halfcheetah_saclag/2025-12-14_20-55_seed7802/history_meanstd.csv",
        "SAC + AugLag":  "../results/HalfCheetah-v3/halfcheetah_sacauglag/2025-12-14_21-00_seed2470/history_meanstd.csv",
        "HJ":           "../results/HalfCheetah-v3/halfcheetah_hjb/2025-12-14_21-30_seed7313/history_meanstd.csv",
        "CAL":           "../results/HalfCheetah-v3/halfcheetah_cal/2025-12-08_21-38_seed2143/history_meanstd.csv",
        "ALGD (Ours)":   "../results/HalfCheetah-v3/halfcheetah_algd/2025-12-09_11-12_seed1148/history_meanstd.csv",
    },
    "Hopper": {
        "SAC + Lag":     "../results/Hopper-v3/hopper_saclag/2025-12-15_00-41_seed7959/history_meanstd.csv",
        "SAC + AugLag":  "../results/Hopper-v3/hopper_sacauglag/2025-12-15_00-55_seed1927/history_meanstd.csv",
        "HJ":           "../results/Hopper-v3/hopper_hjb/2025-12-15_00-58_seed8171/history_meanstd.csv",
        "CAL":           "../results/Hopper-v3/hopper_cal/2025-12-07_07-23_seed2031/history_meanstd.csv",
        # "ALGD (Ours)":   "../results/Hopper-v3/hopper_algd/2025-12-08_03-21_seed1538/history_meanstd.csv",
        "ALGD (Ours)":   "temp_history/hopper_history_meanstd.csv",
    },
    "Ant": {
        "SAC + Lag":     "../results/Ant-v3/ant_saclag/2025-12-16_04-07_seed6873/history_meanstd.csv",
        "SAC + AugLag":  "../results/Ant-v3/ant_sacauglag/2025-12-16_04-38_seed8165/history_meanstd.csv",
        "HJ":           "../results/Ant-v3/ant_hjb/2025-12-16_04-45_seed6818/history_meanstd.csv",
        "CAL":           "../results/Ant-v3/ant_cal/2025-12-10_13-24_seed3809/history_meanstd.csv",
        "ALGD (Ours)":   "../results/Ant-v3/ant_algd/2025-12-11_15-14_seed5428/history_meanstd.csv",
    },
    "Humanoid": {
        "SAC + Lag":     "../results/Humanoid-v3/humanoid_saclag/2025-12-17_00-06_seed8526/history_meanstd.csv",
        "SAC + AugLag":  "../results/Humanoid-v3/humanoid_sacauglag/2025-12-17_01-00_seed6583/history_meanstd.csv",
        "HJ":           "../results/Humanoid-v3/humanoid_hjb/2025-12-16_22-42_seed9964/history_meanstd.csv",
        "CAL":           "../results/Humanoid-v3/humanoid_cal/2025-12-11_14-42_seed6796/history_meanstd.csv",
        # "ALGD (Ours)":   "../results/Humanoid-v3/humanoid_algd/2025-12-14_04-02_seed516/history_meanstd.csv",
        "ALGD (Ours)":   "temp_history/humanoid_history_meanstd.csv",
    },
}

# ======================= 工具函数 =======================
def k_formatter(x, pos):
    x = float(x)
    if abs(x) >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))

def load_history(path):
    """
    读取你现在 safety-gym 同格式的 history_meanstd.csv：
      step, return_mean, return_std_lower, return_std_upper,
            cost_mean,   cost_std_lower,   cost_std_upper
    """
    df = pd.read_csv(path)

    step = df["step"].to_numpy()

    ret_mean  = df["return_mean"].to_numpy()
    ret_upper = df["return_std_upper"].to_numpy()
    ret_lower = df["return_std_lower"].to_numpy()

    cost_mean  = df["cost_mean"].to_numpy()
    cost_upper = df["cost_std_upper"].to_numpy()
    cost_lower = df["cost_std_lower"].to_numpy()

    return step, ret_mean, ret_lower, ret_upper, cost_mean, cost_lower, cost_upper

def safe_exists(path):
    return path is not None and str(path).strip() != "" and os.path.exists(path)

def segment_labels(max_step, n_segments=4):
    edges = np.linspace(0, max_step, n_segments + 1, dtype=int)
    labels = []
    for i in range(n_segments):
        a = edges[i]
        b = edges[i + 1]
        labels.append(f"{int(a/1000)}-{int(b/1000)}k")
    return edges, labels

# ======================= 主绘图 =======================
def main(save_path="mujoco.pdf"):
    # 关键：figsize 与 safety-gym 一致，便于后续拼接（整体宽度一致）
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
        thr = COST_THRESHOLDS.get(f"{task}-v3", None)
        
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

            step, ret_mean, ret_lower, ret_upper, cost_mean, cost_lower, cost_upper = load_history(path)

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

            ax_r.fill_between(
                step_r, ret_lower, ret_upper,
                color=color,
                alpha=OURS_BAND_ALPHA if is_ours else BAND_ALPHA,
                linewidth=0
            )
            line_r, = ax_r.plot(step_r, ret_mean, color=color, linewidth=2, alpha=alpha_line)

            ax_c.fill_between(
                step_c, cost_lower, cost_upper,
                color=color,
                alpha=OURS_BAND_ALPHA if is_ours else BAND_ALPHA,
                linewidth=0
            )
            line_c, = ax_c.plot(step_c, cost_mean, color=color, linewidth=2, alpha=alpha_line)

            if label not in legend_handles:
                legend_handles[label] = line_c

        # 网格、阈值线
        ax_r.grid(True, alpha=0.3)
        ax_c.grid(True, alpha=0.3)
        if thr is not None:
            ax_c.axhline(thr, color="black", linestyle="--", linewidth=1.5, alpha=0.85)

        # x 轴：范围、刻度、k 格式；不显示 xlabel
        for ax in (ax_r, ax_c):
            ax.set_xlim(0, max_step)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(MultipleLocator(tick_step))
            ax.set_xlabel("")

        # y 轴范围（若设了就固定）
        r_ylim = TASK_REWARD_YLIM.get(task, None)
        if r_ylim is not None:
            ax_r.set_ylim(r_ylim[0], r_ylim[1])

        c_ylim = TASK_COST_YLIM.get(task, None)
        if c_ylim is not None:
            ax_c.set_ylim(c_ylim[0], c_ylim[1])

        # y 轴刻度密度
        if task == "Humanoid":
            ax_c.yaxis.set_major_locator(MultipleLocator(5))
            ax_b.yaxis.set_major_locator(MultipleLocator(5))
        else:
            ax_c.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))
            ax_b.yaxis.set_major_locator(MultipleLocator(COST_Y_TICK_STEP))

        r_tick = TASK_REWARD_Y_TICK_STEP.get(task, None)
        if r_tick is not None:
            ax_r.yaxis.set_major_locator(MultipleLocator(r_tick))

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
        ax_b.set_xticklabels(present_labels, rotation=20, ha="right")

        if thr is not None:
            ax_b.axhline(thr, color="black", linestyle="--", linewidth=1.2, alpha=0.85)


        ax_b.legend(
            seg_handles,
            seg_leg_labels,
            loc="best",
            frameon=True,
            fontsize=9,
        )

#     # ====== 顶部统一 legend（黑色圆角矩形框） ======
#     ordered_handles = []
#     ordered_labels = []
#     for lb in MODEL_LABELS:
#         if lb in legend_handles:
#             ordered_handles.append(legend_handles[lb])
#             ordered_labels.append(lb)

#     leg = fig.legend(
#         ordered_handles,
#         ordered_labels,
#         loc="upper center",
#         ncol=len(MODEL_LABELS),
#         frameon=True,
#         fancybox=True,
#         bbox_to_anchor=(0.5, 1.06),
#         borderpad=0.6,
#         handlelength=2.2,
#         columnspacing=1.4,
#     )

#     frame = leg.get_frame()
#     frame.set_edgecolor("grey")
#     frame.set_linewidth(1.5)
#     frame.set_facecolor("white")
#     frame.set_alpha(1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    plt.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")

if __name__ == "__main__":
    main("mujoco.pdf")
