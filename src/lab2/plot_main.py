# import os
# from plot import plot_training_logs
#
# for ph in [name for name in os.listdir('./log')]:
#     log_file = os.path.join('log', ph, 'training_log.csv')
#     plot_training_logs(log_file)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 基础配置
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
sns.set(style='whitegrid', font='Times New Roman')

base_dir = 'log'  # 修改为实际路径

# 获取所有实验目录
exp_dirs = [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]

# 创建复合图表
fig_loss, axs_loss = plt.subplots(3, 4, figsize=(24, 16))
fig_acc, axs_acc = plt.subplots(3, 4, figsize=(24, 16))

# 平铺轴数组方便遍历
axs_loss = axs_loss.flatten()
axs_acc = axs_acc.flatten()

for idx, exp_dir in enumerate(exp_dirs):
    log_path = os.path.join(base_dir, exp_dir, 'training_log.csv')

    if not os.path.exists(log_path):
        continue

    # 数据预处理（与原函数逻辑一致）
    df = pd.read_csv(log_path)
    train_data = df[df['phase'] == 'train'].copy()
    val_data = df[df['phase'] == 'val'].copy()

    n_batches = len(train_data['batch_idx'].unique())
    train_data['global_batch'] = train_data['batch_idx'] + train_data['epoch'] * n_batches
    val_data['global_batch'] = val_data['epoch'] * n_batches + (n_batches - 1)
    combined = pd.concat([train_data, val_data])

    # 绘制loss
    sns.lineplot(x='global_batch', y='loss', hue='phase',
                 data=combined, ax=axs_loss[idx], legend=idx == 0)
    axs_loss[idx].set_title(exp_dir.replace('_', ' '), fontsize=10)
    axs_loss[idx].set_xlabel('')
    axs_loss[idx].set_ylabel('Loss', fontsize=9)

    # 绘制accuracy
    sns.lineplot(x='global_batch', y='accuracy', hue='phase',
                 data=combined, ax=axs_acc[idx], legend=idx == 0)
    axs_acc[idx].set_title(exp_dir.replace('_', ' '), fontsize=10)
    axs_acc[idx].set_xlabel('')
    axs_acc[idx].set_ylabel('Accuracy', fontsize=9)

# 统一格式设置
for fig, title in [(fig_loss, 'Loss Comparison'), (fig_acc, 'Accuracy Comparison')]:
    fig.suptitle(title, fontsize=14, y=0.99)
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.label.set_size(9)
        if ax.get_legend():
            ax.get_legend().remove()

# 添加公共图例
fig_loss.legend(*axs_loss[0].get_legend_handles_labels(),
                loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=8)
fig_acc.legend(*axs_acc[0].get_legend_handles_labels(),
               loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=8)

# plt.tight_layout()
plt.show()
