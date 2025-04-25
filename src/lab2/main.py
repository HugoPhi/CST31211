import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from data_process import get_cifar10_data_loaders
# from models import get_resnet18
from models import SimpleCNN
from training_script import train_model


def main():
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] Using device: {device}')

    # 加载配置
    with open('./config.yml') as f:
        config = yaml.safe_load(f)
        lrs = config['lr']
        batch_sizes = config['batch_size']
        l2s = config['l2']
        augments = config['augment']
        epochs = config['epoch']

    print(f'>> lr: {lrs}')
    print(f'>> bs: {batch_sizes}')
    print(f'>> l2: {l2s}')
    print(f'>> ag: {augments}')
    print(f'>> ep: {epochs}')

    # 加载 CIFAR-10 数据集，每个batch_size, augment对应一个loader -> train, test, axis=0代表aug, axis=1代表bs.
    loaders = [[get_cifar10_data_loaders(batch_size, augment=augment) for batch_size in batch_sizes] for augment in augments]

    def train_once(iag, ibs, epoch, lr, l2):
        print(f'>>> Expriment: use augment={augments[iag]}, batch_size={batch_sizes[ibs]}, lr={lr}, l2={l2}, epochs={epoch}')
        train_loader, test_loader = loaders[iag][ibs]

        '''model replacement'''
        # model = get_resnet18()
        model = SimpleCNN()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=l2, weight_decay=l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

        train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            log_dir=f'augment={augments[iag]}_bs={batch_sizes[ibs]}_lr={lr}_l2={l2}_ep={epoch}',
            num_epochs=epoch,
            resume=False,
        )

    for iag in range(len(augments)):
        for ibs in range(len(batch_sizes)):
            for epoch in epochs:
                for lr in lrs:
                    for l2 in l2s:
                        train_once(
                            iag=iag,
                            ibs=ibs,
                            epoch=epoch,
                            lr=lr,
                            l2=l2,
                        )


if __name__ == '__main__':
    main()
