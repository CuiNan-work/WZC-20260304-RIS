"""
测试文件（带RIS）：加载训练好的PPO模型，运行测试，
生成UAV飞行轨迹图（标注RIS位置）和用户任务处理时延图像。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from stable_baselines3 import PPO

from train import UAVEnv, num_uavs, num_users, ris_pos

# 自动检测系统中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 测试参数
TEST_STEPS = 100
MODEL_PATH = "ppo_uav_ris_10"


def run_test(model_path=MODEL_PATH, test_steps=TEST_STEPS):
    """加载模型并运行测试，返回记录的数据"""

    # 加载模型
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件 '{model_path}.zip' 不存在，请先运行 train.py 训练模型。"
        )

    model = PPO.load(model_path)

    # 创建测试环境
    env = UAVEnv()
    obs, info = env.reset()

    # 数据记录
    # UAV轨迹: (test_steps+1, num_uavs, 2)，包含初始位置
    uav_trajectories = np.zeros((test_steps + 1, num_uavs, 2))
    uav_trajectories[0] = env.uav_positions.copy()

    # 用户时延记录: 通信时延、计算时延、回传时延、总时延
    comm_delays = np.zeros((test_steps, num_users))
    comp_delays = np.zeros((test_steps, num_users))
    return_delays = np.zeros((test_steps, num_users))
    total_delays = np.zeros(test_steps)

    # 用户位置（每步可能不同，记录最后一步用于标注）
    user_positions_record = np.zeros((test_steps, num_users, 2))

    # UAV负载记录: (test_steps, num_uavs)
    uav_loads = np.zeros((test_steps, num_uavs))
    # 用户决策记录: (test_steps, num_users)
    user_decisions_record = np.zeros((test_steps, num_users), dtype=int)

    print(f"\n{'=' * 70}")
    print(f"开始测试（带RIS）：共 {test_steps} 步")
    print(f"{'=' * 70}")

    for step in range(test_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # 记录UAV位置
        uav_trajectories[step + 1] = env.uav_positions.copy()

        # 记录时延
        comm_delays[step] = np.array(env.users_comm_delay)
        comp_delays[step] = np.array(env.users_comp_delay)
        return_delays[step] = np.array(env.users_return_delay)
        total_delays[step] = env.total_time

        # 记录用户位置
        user_positions_record[step] = env.user_positions.copy()

        # 记录UAV负载和用户决策
        uav_loads[step] = np.array(env.uav_L)
        user_decisions_record[step] = env.user_decisions.copy()

        print(f"Step {step + 1:3d} | Reward: {reward:.4f} | "
              f"Total Delay: {env.total_time:.4f}s | "
              f"Decisions: {info['user_decisions']}")

        if done:
            obs, info = env.reset()

    print(f"\n{'=' * 70}")
    print(f"测试完成")
    print(f"{'=' * 70}\n")

    return {
        'uav_trajectories': uav_trajectories,
        'comm_delays': comm_delays,
        'comp_delays': comp_delays,
        'return_delays': return_delays,
        'total_delays': total_delays,
        'user_positions': user_positions_record,
        'uav_loads': uav_loads,
        'user_decisions': user_decisions_record,
        'test_steps': test_steps,
    }


def smooth_line(x, y, num_points=300):
    """使用三次样条插值对折线进行平滑处理"""
    if len(x) < 4:
        return x, y
    x_smooth = np.linspace(x[0], x[-1], num_points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def plot_uav_trajectories(data, save_path="./uav_trajectories_ris.png"):
    """绘制UAV飞行轨迹图（标注RIS位置）"""
    trajectories = data['uav_trajectories']
    user_positions = data['user_positions']
    test_steps = data['test_steps']

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    colors = ['#e74c3c', '#2ecc71', '#3498db']  # 红、绿、蓝
    markers = ['o', 's', '^']

    for i in range(num_uavs):
        xs = trajectories[:, i, 0]
        ys = trajectories[:, i, 1]

        # 绘制轨迹线
        t = np.arange(len(xs))
        _, xs_smooth = smooth_line(t, xs)
        _, ys_smooth = smooth_line(t, ys)
        ax.plot(xs_smooth, ys_smooth, color=colors[i], linewidth=1.5, alpha=0.7,
                label=f'UAV {i + 1} trajectory')

        # 起点标记
        ax.scatter(xs[0], ys[0], color=colors[i], marker=markers[i],
                   s=250, zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(f'UAV{i + 1} start', (xs[0], ys[0]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight='bold', color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # 终点标记
        ax.scatter(xs[-1], ys[-1], color=colors[i], marker='*',
                   s=350, zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(f'UAV{i + 1} end', (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(10, -15),
                    fontsize=10, fontweight='bold', color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # 标注RIS位置（使用ris_pos的x, y坐标）
    ris_x, ris_y = ris_pos[0], ris_pos[1]
    ax.scatter(ris_x, ris_y, color='#9b59b6', marker='D', s=300, zorder=6,
               edgecolors='black', linewidths=2, label=f'RIS (h={ris_pos[2]:.0f}m)')
    ax.annotate(f'RIS\n({ris_x:.0f}, {ris_y:.0f}, h={ris_pos[2]:.0f}m)',
                (ris_x, ris_y),
                textcoords="offset points", xytext=(15, 15),
                fontsize=11, fontweight='bold', color='#9b59b6',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#9b59b6', alpha=0.9))

    # 绘制最后一步的用户位置
    last_user_pos = user_positions[-1]
    ax.scatter(last_user_pos[:, 0], last_user_pos[:, 1],
               color='gray', marker='x', s=80, zorder=4, label='Ground Users')
    for k in range(num_users):
        ax.annotate(f'GT{k}', (last_user_pos[k, 0], last_user_pos[k, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color='gray')

    ax.set_xlim(-420, 420)
    ax.set_ylim(-420, 420)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'UAV Flight Trajectories with RIS ({test_steps} steps)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"UAV飞行轨迹图（含RIS标注）已保存至: {save_path}")


def plot_user_delays(data, save_path="./user_task_delays_ris.png"):
    """绘制用户任务处理时延图像"""
    comm_delays = data['comm_delays']
    comp_delays = data['comp_delays']
    return_delays = data['return_delays']
    total_delays = data['total_delays']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    # === 子图1：各用户通信时延 ===
    ax1 = axes[0, 0]
    for k in range(num_users):
        xs, ys = smooth_line(steps, comm_delays[:, k])
        ys = np.maximum(ys, 0)
        ax1.plot(xs, ys, linewidth=1.2, alpha=0.8, label=f'GT{k}')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Communication Delay (s)', fontsize=11)
    ax1.set_title('Communication Delay per User', fontsize=13)
    ax1.legend(fontsize=7, ncol=2, loc='upper right')
    ax1.set_xticks(range(0, test_steps + 1, 10))
    ax1.set_xlim(0, test_steps + 1)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # === 子图2：各用户计算时延 ===
    ax2 = axes[0, 1]
    for k in range(num_users):
        xs, ys = smooth_line(steps, comp_delays[:, k])
        ys = np.maximum(ys, 0)
        ax2.plot(xs, ys, linewidth=1.2, alpha=0.8, label=f'GT{k}')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Computation Delay (s)', fontsize=11)
    ax2.set_title('Computation Delay per User', fontsize=13)
    ax2.legend(fontsize=7, ncol=2, loc='upper right')
    ax2.set_xticks(range(0, test_steps + 1, 10))
    ax2.set_xlim(0, test_steps + 1)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    # === 子图3：各用户回传时延 ===
    ax3 = axes[1, 0]
    for k in range(num_users):
        xs, ys = smooth_line(steps, return_delays[:, k])
        ys = np.maximum(ys, 0)
        ax3.plot(xs, ys, linewidth=1.2, alpha=0.8, label=f'GT{k}')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Return Delay (s)', fontsize=11)
    ax3.set_title('Return Delay per User', fontsize=13)
    ax3.legend(fontsize=7, ncol=2, loc='upper right')
    ax3.set_xticks(range(0, test_steps + 1, 10))
    ax3.set_xlim(0, test_steps + 1)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    # === 子图4：系统总时延 ===
    ax4 = axes[1, 1]
    xs, ys = smooth_line(steps, total_delays)
    ys = np.maximum(ys, 0)
    ax4.plot(xs, ys, 'r-', linewidth=2, label='Total System Delay')
    ax4.fill_between(xs, 0, ys, alpha=0.2, color='red')
    ax4.set_xlabel('Step', fontsize=11)
    ax4.set_ylabel('Total System Delay (s)', fontsize=11)
    ax4.set_title('Total System Delay per Step', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.set_xticks(range(0, test_steps + 1, 10))
    ax4.set_xlim(0, test_steps + 1)
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'User Task Processing Delays with RIS ({test_steps} steps)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"用户任务处理时延图已保存至: {save_path}")


def plot_jain_fairness_index(data, save_path="./jain_fairness_index_ris.png"):
    """绘制每一步UAV负载的Jain公平指数"""
    uav_loads = data['uav_loads']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    # 计算每步的Jain公平指数: J = (Σxi)^2 / (n * Σxi^2)
    sum_x = uav_loads.sum(axis=1)
    sum_x2 = (uav_loads ** 2).sum(axis=1)
    # 避免除以零
    jain_index = np.where(sum_x2 > 0, sum_x ** 2 / (num_uavs * sum_x2), 1.0)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    xs, ys = smooth_line(steps, jain_index)
    ax.plot(xs, ys, color='#2980b9', linewidth=2, label="Jain's Fairness Index")
    ax.fill_between(xs, 0, ys, alpha=0.15, color='#2980b9')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Perfect Fairness (1.0)')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax.set_title(f"Jain's Fairness Index of UAV Load per Step with RIS ({test_steps} steps)", fontsize=14)
    ax.set_xticks(range(0, test_steps + 1, 10))
    ax.set_xlim(0, test_steps + 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Jain公平指数图已保存至: {save_path}")


def plot_offload_distribution(data, save_path="./offload_distribution_ris.png"):
    """绘制每一步所有用户的本地计算和卸载计算分布"""
    user_decisions = data['user_decisions']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    # 统计每步中 本地计算用户数 和 卸载到各UAV的用户数
    local_counts = np.zeros(test_steps)
    uav_counts = np.zeros((test_steps, num_uavs))

    for t in range(test_steps):
        for k in range(num_users):
            dec = user_decisions[t, k]
            if dec == 0:
                local_counts[t] += 1
            else:
                uav_counts[t, dec - 1] += 1

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # 折线图：每类决策的用户数随步骤变化
    line_data = [local_counts] + [uav_counts[:, i] for i in range(num_uavs)]
    line_labels = ['Local'] + [f'Offload to UAV {i + 1}' for i in range(num_uavs)]
    line_colors = ['#95a5a6'] + ['#e74c3c', '#2ecc71', '#3498db'][:num_uavs]
    all_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    line_markers = all_markers[:1 + num_uavs]

    for i, (yd, label, color, marker) in enumerate(zip(line_data, line_labels, line_colors, line_markers)):
        xs, ys = smooth_line(steps, yd)
        ys = np.maximum(ys, 0)
        ax.plot(xs, ys, color=color, linewidth=2, label=label,
                marker=marker, markevery=max(1, test_steps // 10),
                markersize=6, alpha=0.9)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title(f'User Computation Distribution per Step with RIS ({test_steps} steps)', fontsize=14)
    ax.set_xticks(range(0, test_steps + 1, 10))
    ax.set_xlim(0, test_steps + 1)
    ax.set_yticks(range(0, num_users + 1))
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"用户计算分布图已保存至: {save_path}")


def plot_local_vs_offload_delay(data, save_path="./local_vs_offload_delay_ris.png"):
    """绘制每一步本地时延和卸载时延的对比折线图"""
    user_decisions = data['user_decisions']
    comm_delays = data['comm_delays']
    comp_delays = data['comp_delays']
    return_delays = data['return_delays']
    test_steps = data['test_steps']

    steps = np.arange(1, test_steps + 1)

    # 每步的本地总时延 和 卸载总时延
    local_delay_per_step = np.zeros(test_steps)
    offload_delay_per_step = np.zeros(test_steps)

    for t in range(test_steps):
        for k in range(num_users):
            user_total = comm_delays[t, k] + comp_delays[t, k] + return_delays[t, k]
            if user_decisions[t, k] == 0:
                local_delay_per_step[t] += user_total
            else:
                offload_delay_per_step[t] += user_total

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    xs1, ys1 = smooth_line(steps, local_delay_per_step)
    xs2, ys2 = smooth_line(steps, offload_delay_per_step)
    ys1 = np.maximum(ys1, 0)
    ys2 = np.maximum(ys2, 0)
    ax.plot(xs1, ys1, '-', color='#e67e22', linewidth=2,
            label='Local Computation Delay')
    ax.plot(xs2, ys2, '-', color='#2980b9', linewidth=2,
            label='Offload Computation Delay')

    ax.fill_between(xs1, ys1, alpha=0.15, color='#e67e22')
    ax.fill_between(xs2, ys2, alpha=0.15, color='#2980b9')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Total Delay (s)', fontsize=12)
    ax.set_title(f'Local vs Offload Delay per Step with RIS ({test_steps} steps)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(range(0, test_steps + 1, 10))
    ax.set_xlim(0, test_steps + 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"本地与卸载时延对比图已保存至: {save_path}")


if __name__ == "__main__":
    data = run_test(model_path=MODEL_PATH, test_steps=TEST_STEPS)
    plot_uav_trajectories(data, save_path="./uav_trajectories_ris.png")
    plot_user_delays(data, save_path="./user_task_delays_ris.png")
    plot_jain_fairness_index(data, save_path="./jain_fairness_index_ris.png")
    plot_offload_distribution(data, save_path="./offload_distribution_ris.png")
    plot_local_vs_offload_delay(data, save_path="./local_vs_offload_delay_ris.png")
