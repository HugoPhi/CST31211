import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

from models import SimpleCNN
from data_process import get_cifar10_data_loaders

ix2name = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def plot_training_logs(log_file):
    '''
    根据 training_log.csv 文件绘制 train_loss, val_loss vs. batch 以及 train_acc, val_acc vs. batch。

    参数:
        log_file: 日志文件路径 (例如 'log/training_log.csv')。
    '''
    # 设置全局字体为 Times New Roman
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']

    # 设置Seaborn样式
    sns.set(style='whitegrid', font='Times New Roman')

    # 检查日志文件是否存在
    if not os.path.exists(log_file):
        print(f'Log file {log_file} does not exist.')
        return

    # 加载日志文件到 DataFrame
    df = pd.read_csv(log_file)

    # 筛选出训练和验证阶段的数据
    train_data = df[df['phase'] == 'train'].copy()
    val_data = df[df['phase'] == 'val'].copy()

    # 计算全局 batch 编号
    train_data['global_batch'] = train_data['batch_idx'] + train_data['epoch'] * len(train_data['batch_idx'].unique())

    # 验证数据的 global_batch 应该对应于每个 epoch 的最后一个训练 batch
    val_data['global_batch'] = val_data['epoch'] * len(train_data['batch_idx'].unique()) + len(train_data['batch_idx'].unique()) - 1

    # 合并数据以便于绘图
    combined_data = pd.concat([train_data, val_data], ignore_index=True)

    # 创建绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制 loss 曲线
    sns.lineplot(x='global_batch', y='loss', hue='phase', data=combined_data, ax=axes[0])
    axes[0].set_title('Loss vs. Batch', fontsize=14, fontname='Times New Roman')
    axes[0].set_xlabel('Global Batch Index', fontsize=12, fontname='Times New Roman')
    axes[0].set_ylabel('Loss', fontsize=12, fontname='Times New Roman')

    # 绘制 accuracy 曲线
    sns.lineplot(x='global_batch', y='accuracy', hue='phase', data=combined_data, ax=axes[1])
    axes[1].set_title('Accuracy vs. Batch', fontsize=14, fontname='Times New Roman')
    axes[1].set_xlabel('Global Batch Index', fontsize=12, fontname='Times New Roman')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontname='Times New Roman')

    # 设置刻度字体
    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(10)

    # 设置图例字体
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontname('Times New Roman')
                text.set_fontsize(10)

    # 调整布局
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_loader, device):
    '''
    绘制混淆矩阵，并保存到指定路径。

    参数:
        model: 已加载的 PyTorch 模型。
        test_loader: 测试数据集的 DataLoader。
        device: 使用的设备（'cpu' 或 'cuda'）。
    '''
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 获取预测类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.set(style='whitegrid', font='Times New Roman')
    cmap = plt.cm.Greens  # 使用绿色调
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                     xticklabels=ix2name, yticklabels=ix2name)

    # 设置标题和标签
    ax.set_title('Confusion Matrix', fontsize=14, fontname='Times New Roman')
    ax.set_xlabel('Predicted Label', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('True Label', fontsize=12, fontname='Times New Roman')

    # 设置刻度字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)

    # 调整布局
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    log_dir = 'log/log_bs=64_lr=0.01_ep=10'

    if False:
        log_file = os.path.join(log_dir, 'training_log.csv')
        plot_training_logs(log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_loader = get_cifar10_data_loaders(64)

    model_path = os.path.join(log_dir, 'checkpoints', 'best_model.pth')  # 替换为你的模型路径
    model = SimpleCNN().to(device)  # 根据你的模型参数初始化

    # 加载检查点文件
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    plot_confusion_matrix(model, test_loader, device)
