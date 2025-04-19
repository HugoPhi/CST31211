import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from data_process import get_cifar10_data_loaders  # 假设数据加载函数在单独的文件中
from models import SimpleCNN


# 定义训练函数
def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    log_dir,
    device,
    num_epochs=25,
    resume=False,
):
    '''
    训练模型并支持检查点保存和日志记录。

    参数:
        model: 要训练的模型。
        criterion: 损失函数。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        train_loader: 训练数据加载器。
        test_loader: 测试数据加载器。
        log_dir: 在log下的路径，用来存放这次实验的所有日志以及检查点。
        device: 设备。
        num_epochs: 训练的总轮数，默认为 25。
        resume: 是否从检查点恢复训练，默认为 False。
    '''

    # 创建本次运行的日志目录
    os.makedirs(os.path.join('log', log_dir), exist_ok=True)

    chkp_dir = os.path.join('log', log_dir, 'checkpoints')
    log_file = os.path.join('log', log_dir, 'training_log.csv')

    # 创建检查点目录
    os.makedirs(chkp_dir, exist_ok=True)

    # 初始化日志文件
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'phase', 'batch_idx', 'loss', 'accuracy'])

    best_acc = 0.0  # 用于记录最佳准确率
    start_epoch = 0

    # 如果 resume 为 True，则加载检查点
    if resume:
        checkpoint_path = os.path.join(chkp_dir, 'best_model.pth')  # 加载最佳模型检查点
        if os.path.exists(checkpoint_path):
            print(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            print(f'Resuming training from epoch {start_epoch} with best accuracy {best_acc:.4f}')
        else:
            print('No checkpoint found. Starting training from scratch.')

    for epoch in range(start_epoch, num_epochs):
        print(f'@ Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = test_loader

            running_loss = 0.0
            running_corrects = 0
            epoch_logs = []

            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和正确预测数量
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 计算当前 batch 的损失和准确率
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).item() / inputs.size(0)

                # 记录日志
                epoch_logs.append([epoch + 1, phase, batch_idx, batch_loss, batch_acc])

            # 计算 epoch 的损失和准确率
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            print(f'[*] {phase.capitalize():<10} Loss: {epoch_loss:.4f}, {phase.capitalize():<10} Acc: {epoch_acc:.4f}')

            # 将日志写入文件
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(epoch_logs)

        # 更新学习率
        scheduler.step()

        # 保存当前 epoch 的模型权重
        epoch_chkp_path = os.path.join(chkp_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }, epoch_chkp_path)
        # print(f'[!] Model weights saved at {epoch_chkp_path}')

        # 如果是验证阶段且准确率更高，则保存最佳模型
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_chkp_path = os.path.join(chkp_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }, best_chkp_path)
            print(f'[!] Best model saved at {best_chkp_path} with accuracy {best_acc:.4f}')

        print()


# 主程序
if __name__ == '__main__':
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # hyper params
    batch_size = 64
    lr = 0.01
    epochs = 10

    # 加载 CIFAR-10 数据集
    train_loader, test_loader = get_cifar10_data_loaders(batch_size, augment=True)

    # 定义模型
    model = SimpleCNN()  # 不使用预训练权重
    model = model.to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练模型
    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        log_dir=f'test/log_bs={batch_size}_lr={lr}_ep={epochs}',
        num_epochs=epochs,
        resume=True,
    )
