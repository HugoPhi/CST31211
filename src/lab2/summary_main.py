import os
import pandas as pd


def calculate_epoch_metrics(base_dir='log'):
    # 结果容器
    results = []

    # 遍历实验目录
    for exp_dir in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, exp_dir)

        if not os.path.isdir(exp_path):
            continue

        log_file = os.path.join(exp_path, 'training_log.csv')
        if not os.path.exists(log_file):
            print(f"跳过缺失日志的目录: {exp_dir}")
            continue

        try:
            df = pd.read_csv(log_file)
        except Exception as e:
            print(f"读取 {log_file} 失败: {e}")
            continue

        # 处理训练数据
        train_df = df[df['phase'] == 'train']

        if not train_df.empty:
            # 按epoch取平均
            epoch_train = train_df.groupby('epoch').agg({
                'accuracy': 'mean',
                'loss': 'mean'
            }).reset_index()
            max_train_acc = epoch_train['accuracy'].max()
            max_train_loss = epoch_train['loss'].max()
        else:
            max_train_acc = max_train_loss = None

        # 处理验证数据
        val_df = df[df['phase'] == 'val']
        if not val_df.empty:
            # 按epoch取平均
            epoch_val = val_df.groupby('epoch').agg({
                'accuracy': 'mean',
                'loss': 'mean'
            }).reset_index()
            max_val_acc = epoch_val['accuracy'].max()
            max_val_loss = epoch_val['loss'].max()
        else:
            max_val_acc = max_val_loss = None

        # 记录结果
        results.append({
            '实验名称': exp_dir,
            '最大训练准确率': f"{max_train_acc:.4f}" if max_train_acc else "N/A",
            '最大训练损失': f"{max_train_loss:.4f}" if max_train_loss else "N/A",
            '最大验证准确率': f"{max_val_acc:.4f}" if max_val_acc else "N/A",
            '最大验证损失': f"{max_val_loss:.4f}" if max_val_loss else "N/A"
        })

    # 格式化输出
    print("\n各实验按epoch平均后的最大指标汇总:")
    print(f"{'实验名称':<50} | {'Train Acc':<8} | {'Train Loss':<8} | {'Val Acc':<8} | {'Val Loss':<8}")
    print("-" * 95)
    for res in results:
        print(f"{res['实验名称']:<50} | "
              f"{res['最大训练准确率']:>8} | "
              f"{res['最大训练损失']:>8} | "
              f"{res['最大验证准确率']:>8} | "
              f"{res['最大验证损失']:>8}")


if __name__ == '__main__':
    calculate_epoch_metrics(base_dir='log')  # 修改为实际路径
