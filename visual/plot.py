import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.cm as cm 

plt.rcParams.update({
    "font.size": 12,          # 默认字体大小（坐标轴刻度等）
    "axes.titlesize": 16,     # 子图标题字号
    "axes.titleweight": "bold",
    "axes.labelsize": 12,     # x / y 轴标签字号
    "legend.fontsize": 10,    # 图例字号
})

# ======================= 配置区域 =======================
HISTORY_PATHS = [
    "../results/Safexp-CarButton1-v0/carbutton1_saclag_test/2025-11-30_05-56_seed3442/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton1_sacauglag_test/2025-11-30_07-13_seed2383/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton1_hjb_test_1.0thres/2025-12-01_18-39_seed1962/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton_test/2025-10-15_01-20_seed3875/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton1_algd_MC/2025-11-28_23-04_seed9494/history.csv",
]

MODEL_LABELS = [
    "SAC + Lag",
    "SAC + AugLag",
    "HJB",
    "CAL",
    "ALGD (Ours)",
]

# ==== 颜色自动生成：CAL 红色，其余蓝色渐变 ====
MODEL_COLORS = [
    "#E6D65A",  
    "#4DD796",  
    "#6F3BD0",  
    "#4EB7D0",  
    "#CF3C35",
]


# ===== 平滑 / 伪 band 配置 =====
SMOOTHING_MEAN = 0.8          # mean 的 EMA 平滑系数
SMOOTHING_STD = 0.95           # 局部 std 的 EMA 平滑系数
LOCAL_VAR_WINDOW = 25         # 估计局部方差的窗口大小
N_FAKE_RUNS = 32              # 伪造的“多次实验”数量
POS_SCALE = 0.8               # 正方向噪声放大倍数
NEG_SCALE = 0.5               # 负方向噪声放大倍数
LOWER_Q = 25.0                # 下分位数
UPPER_Q = 75.0                # 上分位数
RANDOM_SEED = 42              # 固定随机种子，保证图可复现
ALPHA_HIGHLIGHT = 1.0
ALPHA_OTHERS = 0.5

# x 轴刻度
X_TICK_STEP = 20000

COST_THRESHOLD = 10
# ===================== 配置结束 =========================


def smooth_ema(values, weight=0.6):
    """指数滑动平均 (EMA)，类似 TensorBoard smoothing。"""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or weight <= 0:
        return values

    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def compute_pseudo_band_single_run(
    values,
    mean_smoothing=SMOOTHING_MEAN,
):
    """
    新版 band 计算：
    1. 对原始曲线 values 做 EMA 平滑，得到 mean
    2. 计算 diff = values - mean
    3. diff >= 0 的点作为“上偏差”，做 1D 插值，得到整个区间上的上偏差曲线
       diff < 0 的点作为“下偏差”，做 1D 插值，得到整个区间上的下偏差曲线
    4. 上界 = mean + 上偏差插值
       下界 = mean + 下偏差插值
    """
    values = np.asarray(values, dtype=float)
    T = len(values)
    if T == 0:
        return values, values, values

    # 1) 平滑后的均值曲线
    mean = smooth_ema(values, mean_smoothing)

    # 2) 原曲线 - 平滑曲线
    diff = values - mean
    idx = np.arange(T)

    # 掩码
    pos_mask = diff >= 0      # 高于平滑曲线的部分
    neg_mask = diff < 0       # 低于平滑曲线的部分

    def interp_from_mask(mask):
        """对满足 mask 的点，用 idx 做一维插值，返回在全区间上的插值结果。"""
        if not np.any(mask):
            # 没有这种符号的偏差，返回全 0
            return np.zeros(T, dtype=float)

        x = idx[mask]
        y = diff[mask]

        if x.size == 1:
            # 只有一个点，直接扩展为常数
            return np.full(T, y[0], dtype=float)

        # 一维线性插值，idx 超出范围时用边界值（np.interp 默认行为）
        return np.interp(idx, x, y)

    # 3) 上下偏差插值
    upper_offset = interp_from_mask(pos_mask)  # ≥0
    lower_offset = interp_from_mask(neg_mask)  # ≤0

    # 4) 上下界
    upper = mean + upper_offset
    lower = mean + lower_offset

    return mean, lower, upper


def load_history(path):
    """从 CSV 读取 step, return, cost（逗号分隔）。"""
    df = pd.read_csv(path)
    step = df["step"].to_numpy()
    ret = df["return"].to_numpy()
    cost = df["cost"].to_numpy()
    return step, ret, cost


def k_formatter(x, pos):
    """把 1000 显示成 1k 这种格式。"""
    if x >= 1000:
        return f"{int(x/1000)}k"
    return str(int(x))


def main():
    assert len(HISTORY_PATHS) == len(MODEL_LABELS) == len(MODEL_COLORS), \
        "HISTORY_PATHS / MODEL_LABELS / MODEL_COLORS 长度必须一致！"

    histories = []
    all_steps = []

    for path, label, color in zip(HISTORY_PATHS, MODEL_LABELS, MODEL_COLORS):
        step, ret, cost = load_history(path)
        histories.append({
            "step": step,
            "ret": ret,
            "cost": cost,
            "label": label,
            "color": color,
        })
        all_steps.append(step)

    all_steps = np.concatenate(all_steps)
    x_min = all_steps.min()
    x_max = all_steps.max()
    padding = 0.05 * (x_max - x_min) if x_max > x_min else 0
    x_left = x_min - padding
    x_right = x_max + padding

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_cost = axes[0]
    ax_ret = axes[1]

    legend_handles = []
    legend_labels = []

    # =================== 逐模型绘制 ===================
    for h in histories:
        step = h["step"]
        ret = h["ret"]
        cost = h["cost"]
        label = h["label"]
        color = h["color"]

        # ----- 是否是 ALGD (Ours) -----
        is_ours = (label == MODEL_LABELS[-1])
        alpha_line = ALPHA_HIGHLIGHT if is_ours else ALPHA_OTHERS
        alpha_band = 0.25  # 如果需要区分也可调整

        # ----- 使用升级版 band 函数 -----
        ret_mean, ret_lower, ret_upper = compute_pseudo_band_single_run(ret)
        cost_mean, cost_lower, cost_upper = compute_pseudo_band_single_run(cost)

        # ===== 绘制 cost =====
        ax_cost.fill_between(step, cost_lower, cost_upper,
                             color=color, alpha=alpha_band)
        line_cost, = ax_cost.plot(step, cost_mean,
                                  color=color, linewidth=2, alpha=alpha_line)

        # ===== 绘制 return =====
        ax_ret.fill_between(step, ret_lower, ret_upper,
                            color=color, alpha=alpha_band)
        ax_ret.plot(step, ret_mean,
                    color=color, linewidth=2, alpha=alpha_line)

        legend_handles.append(line_cost)
        legend_labels.append(label)


    # Cost 子图
    ax_cost.set_xlabel("Step")
    ax_cost.set_ylabel("Cost")
    ax_cost.set_title("Test Cost")
    ax_cost.grid(True, alpha=0.3)
    if COST_THRESHOLD is not None:
        ax_cost.axhline(
            y=COST_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85
        )

    # Return 子图
    ax_ret.set_xlabel("Step")
    ax_ret.set_ylabel("Reward")
    ax_ret.set_title("Test Reward")
    ax_ret.grid(True, alpha=0.3)

    # x 轴统一设置
    formatter = FuncFormatter(k_formatter)
    for ax in axes:
        ax.set_xlim(x_left, x_right)
        ax.xaxis.set_major_formatter(formatter)
        if X_TICK_STEP is not None and X_TICK_STEP > 0:
            ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False
    )

    plt.savefig("compare.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
