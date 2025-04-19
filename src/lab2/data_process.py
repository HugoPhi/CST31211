from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_data_loaders(batch_size, root='~/.torch/datasets', download=True, augment=False):
    '''
    获取 CIFAR-10 数据集的训练和测试数据加载器，支持数据增强。

    参数:
        batch_size (int): 批量大小。
        root (str): 数据集存储的根目录，默认为 '~/.torch/datasets'。
        download (bool): 是否下载数据集，默认为 True。
        augment (bool): 是否启用数据增强，默认为 False。

    返回:
        train_loader (DataLoader): 训练数据加载器。
        test_loader (DataLoader): 测试数据加载器。
    '''

    # 测试集的预处理（始终不变）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 训练集的预处理
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        download=download
    )

    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=test_transform,
        download=download
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


# 示例使用
if __name__ == '__main__':
    batch_size = 64
    augment = True  # 启用数据增强
    train_loader, test_loader = get_cifar10_data_loaders(batch_size, augment=augment)

    # 验证数据加载器
    for images, labels in train_loader:
        print(f'Batch of images shape: {images.shape}')
        print(f'Batch of labels shape: {labels.shape}')
        break
