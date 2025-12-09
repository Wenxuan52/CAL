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
    "legend.fontsize": 14,    # 图例字号
})

# ======================= 配置区域 =======================
# 任意数量的模型路径
HISTORY_PATHS = [
    "../results/Safexp-CarButton1-v0/carbutton_test/2025-10-15_01-20_seed3875/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton1_hjb_test_0.5thres/2025-12-01_18-52_seed9485/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton1_hjb_test_1.0thres/2025-12-01_18-39_seed1962/history.csv",
    "../results/Safexp-CarButton1-v0/carbutton1_hjb_test_10thres/2025-12-01_08-12_seed6683/history.csv",
]

# 对应的模型名字（用于 legend）
MODEL_LABELS = [
    "CAL",
    "HJB threshold = 0.5",
    "HJB threshold = 1.0",
    "HJB threshold = 10.0",
]

# 对应的颜色
# ==== 颜色自动生成：CAL 红色，其余蓝色渐变 ====
CAL_COLOR = "tab:red"
num_models = len(HISTORY_PATHS)

# 用 matplotlib 的 "Blues" colormap 生成 N-1 个蓝色，从浅到深
blue_cmap = cm.get_cmap("Blues")
blue_levels = np.linspace(0.3, 0.9, num_models - 1)  # 0.3~0.9 避免太浅/太深

BLUE_GRADIENT_COLORS = [blue_cmap(lvl) for lvl in blue_levels]

# 第一个是 CAL（红色），后面全是蓝色渐变
MODEL_COLORS = [CAL_COLOR] + BLUE_GRADIENT_COLORS



SMOOTHING = 0.7  # 0~1，越大越平滑
SMOOTHING_std = 0.99

# x 轴刻度间隔（比如 20000 代表 0, 20k, 40k ...）
X_TICK_STEP = 20000

# Cost 的安全阈值线（不需要可以设为 None）
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
    # 一些简单的检查：三个列表长度必须一致
    assert len(HISTORY_PATHS) == len(MODEL_LABELS) == len(MODEL_COLORS), \
        "HISTORY_PATHS / MODEL_LABELS / MODEL_COLORS 长度必须一致！"

    # 1. 读取所有模型的数据
    histories = []  # 每个元素: dict(step, ret, cost, label, color)
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

    # 把所有 step 拼起来用于自动设置 x 轴范围
    all_steps = np.concatenate(all_steps)
    x_min = all_steps.min()
    x_max = all_steps.max()
    # 给左右各留 5% 的边距
    padding = 0.05 * (x_max - x_min) if x_max > x_min else 0
    x_left = x_min - padding
    x_right = x_max + padding

    # 2. 创建画布：一行两列
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_cost = axes[0]
    ax_ret = axes[1]

    legend_handles = []
    legend_labels = []

    # 3. 逐个模型画曲线（加入平滑后的 std 绘制）
    for h in histories:
        step = h["step"]
        ret = h["ret"]
        cost = h["cost"]
        label = h["label"]
        color = h["color"]

        # =============================
        # 1. 平滑后的 mean
        # =============================
        ret_mean = smooth_ema(ret, SMOOTHING)
        cost_mean = smooth_ema(cost, SMOOTHING)

        # =============================
        # 2. raw std = |raw - mean|
        #    然后对 std 再做一次平滑 (关键！)
        # =============================
        ret_std_raw = np.abs(ret - ret_mean)
        cost_std_raw = np.abs(cost - cost_mean)

        ret_std = smooth_ema(ret_std_raw, SMOOTHING_std)
        cost_std = smooth_ema(cost_std_raw, SMOOTHING_std)

        # 上下界
        ret_upper = ret_mean + ret_std
        ret_lower = ret_mean - ret_std

        cost_upper = cost_mean + cost_std
        cost_lower = cost_mean - cost_std

        # =============================
        # 绘制 cost（含平滑误差带）
        # =============================
        # 上下界（浅色）
        ax_cost.plot(step, cost_upper, color=color, alpha=0.01, linewidth=1)
        ax_cost.plot(step, cost_lower, color=color, alpha=0.01, linewidth=1)

        # 填充区域（更淡）
        ax_cost.fill_between(step, cost_lower, cost_upper, color=color, alpha=0.2)

        # mean（主曲线）
        line_cost, = ax_cost.plot(step, cost_mean, color=color, linewidth=2, alpha=1)

        # =============================
        # 绘制 return（含平滑误差带）
        # =============================
        ax_ret.plot(step, ret_upper, color=color, alpha=0.01, linewidth=1)
        ax_ret.plot(step, ret_lower, color=color, alpha=0.01, linewidth=1)

        ax_ret.fill_between(step, ret_lower, ret_upper, color=color, alpha=0.2)

        ax_ret.plot(step, ret_mean, color=color, linewidth=2, alpha=1)

        # legend 使用 cost 的 mean 即可
        legend_handles.append(line_cost)
        legend_labels.append(label)


    # 4. 子图统一设置

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

    # x 轴统一设置：自动范围 + k 格式 + 固定刻度间隔
    formatter = FuncFormatter(k_formatter)
    for ax in axes:
        ax.set_xlim(x_left, x_right)
        ax.xaxis.set_major_formatter(formatter)
        if X_TICK_STEP is not None and X_TICK_STEP > 0:
            ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))

    # 5. 统一 legend 放在整张图最下面
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 底部留 5% 高度给 legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(len(legend_labels), 4),  # 一行最多 4 个，更多就自动换行
        frameon=False
    )

    plt.savefig("models_compare.pdf", bbox_inches="tight")
    # 如果想看图就取消注释
    # plt.show()


if __name__ == "__main__":
    main()
