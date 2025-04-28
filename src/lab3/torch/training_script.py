import collections
import os
import yaml
import torch
import logging
import threading
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import os
import tempfile


from data_process import src_vocab, trg_vocab, train_loader, valid_loader

from model import Encoder, Decoder, Transformer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'log/training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)


class TransformerTrainer:
    def __init__(self, config):
        self.config = config

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device
        print(f"using device: {device}")

        self.model = Transformer(
            encoder=Encoder(
                src_vocab_size=len(src_vocab.word2idx),
                embedding_size=config['d_model'],
                head_size=config['n_head'],
                ffn_size=config['ffn_size'],
                num_blocks=config['num_blocks'],
                p=config['dropout']
            ),
            decoder=Decoder(
                trg_vocab_size=len(trg_vocab.word2idx),
                embedding_size=config['d_model'],
                head_size=config['n_head'],
                ffn_size=config['ffn_size'],
                num_blocks=config['num_blocks'],
                p=config['dropout']
            )
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['lr'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config['weight_decay']
        )

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) ** -0.5,
                (step + 1) * (config['warmup_steps'] ** -1.5)
            )
        )

        # 混合精度训练
        self.scaler = torch.amp.GradScaler(enabled=config['use_amp'])

        # 损失函数（忽略padding）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        os.makedirs(config['output_dir'], exist_ok=True)

        # 新增训练状态跟踪
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 智能恢复训练逻辑
        self._auto_resume_training()



    def _prepare_batch(self, batch):
        """处理原始批次数据"""
        src_batch, trg_batch = batch
        src_tensor = torch.tensor(src_batch).to(self.device)
        trg_tensor = torch.tensor(trg_batch).to(self.device)

        # 生成decoder输入输出
        trg_input = trg_tensor[:, :-1]  # 移除最后一个token
        trg_output = trg_tensor[:, 1:]  # 移除第一个token

        return {
            'src': src_tensor,
            'trg_input': trg_input,
            'trg_output': trg_output,
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            prepared_batch = self._prepare_batch(batch)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logits = self.model(
                    encoder_input=prepared_batch['src'],
                    decoder_input=prepared_batch['trg_input'],
                    src_who_is_pad=0,
                    trg_who_is_pad=0
                )

                loss = self.criterion(
                    logits.view(-1, len(trg_vocab.idx2word)),
                    prepared_batch['trg_output'].contiguous().view(-1)
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        for batch in val_loader:
            prepared_batch = self._prepare_batch(batch)

            logits = self.model(
                encoder_input=prepared_batch['src'],
                decoder_input=prepared_batch['trg_input'],
                src_who_is_pad=0,
                trg_who_is_pad=0
            )

            loss = self.criterion(
                logits.view(-1, len(trg_vocab.idx2word)),
                prepared_batch['trg_output'].contiguous().view(-1)
            )
            total_loss += loss.item()

        return total_loss / len(val_loader)

    def _auto_resume_training(self):
        """智能恢复训练状态的核心逻辑"""
        output_dir = self.config['output_dir']
        latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
        best_path = os.path.join(output_dir, "best_model.pt")

        # 条件判断优先级：强制训练 > 指定检查点 > 自动恢复
        if self.config.get('force_retrain'):
            logging.info("强制重新训练模式，忽略所有检查点")
            return

        if self.config.get('resume_checkpoint'):
            if self.config['resume_checkpoint'].lower() == "best":
                checkpoint_path = best_path
            elif self.config['resume_checkpoint'].lower() == "latest":
                checkpoint_path = latest_path
            else:
                checkpoint_path = self.config['resume_checkpoint']
            
            if os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)

                if self.current_epoch >= self.config['num_epochs']:
                    logging.warning("检查点epoch超过配置总数，重置训练状态")
                    self._reset_training()
                return
            else:
                logging.warning(f"指定检查点不存在: {checkpoint_path}")

        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path, map_location=self.device)
            if checkpoint['epoch'] >= self.config['num_epochs']:
                logging.info("发现已完成训练的检查点，启用新训练周期")
                self._reset_training()
            else:
                logging.info("发现未完成训练的检查点，自动恢复")
                self._load_checkpoint(latest_path)
        else:
            logging.info("未找到检查点，开始新训练")

    def _reset_training(self):
        """重置所有训练状态"""
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.model.apply(self._init_weights)  # 假设有参数初始化方法
        self.optimizer.state = collections.defaultdict(dict)
        self.lr_scheduler.last_epoch = -1
        self.scaler = torch.amp.GradScaler(enabled=self.config['use_amp'])

    def _init_weights(self, module):
        """参数初始化（示例）"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def _load_checkpoint(self, checkpoint_path):
        """加载检查点的内部实现"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logging.info(f"成功加载检查点：{checkpoint_path} (epoch {self.current_epoch})")

    def save_checkpoint(self, is_best=False):
        """智能保存检查点"""
        state = {
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict(),
            'config': self.config
        }

        # 始终保存最新检查点
        latest_path = os.path.join(self.config['output_dir'], "checkpoint_latest.pt")
        torch.save(state, latest_path)
        logging.debug(f"更新最新检查点：{latest_path}")

        # 保存最佳模型（不覆盖）
        if is_best:
            best_path = os.path.join(self.config['output_dir'], "best_model.pt")
            torch.save(state, best_path)
            logging.info(f"发现新最佳模型：loss {state['best_loss']:.4f}")

    def train(self, train_loader, val_loader=None):
        try:
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                self.current_epoch = epoch + 1  # 更新当前epoch
                logging.info(f"训练周期 [{self.current_epoch}/{self.config['num_epochs']}]")

                # 训练与验证流程
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate(valid_loader) if val_loader else float('inf')
                # print(f'[*] train loss: {train_loss}, val loss: {val_loss}, best val loss: {self.best_loss}')
                logging.info(f'train loss: {train_loss}, val loss: {val_loss}, best val loss: {self.best_loss}')

                # 更新最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(is_best=True)

                # 智能保存策略
                if self.current_epoch % self.config['save_interval'] == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            logging.info(f"Training interrupted while runnig epoch {self.current_epoch}.")


if __name__ == "__main__":
    with open('./config.yml') as f:
        config = yaml.safe_load(f)['train']

    trainer = TransformerTrainer(config)

    try:
        trainer.train(train_loader, valid_loader)
    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint(epoch='interrupted')
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
