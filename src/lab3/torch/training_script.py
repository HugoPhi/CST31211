import os
import yaml
import torch
import logging
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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

    def _prepare_batch(self, batch):
        """处理原始批次数据"""
        src_batch, trg_batch = batch
        src_tensor = torch.tensor(src_batch).to(self.device)
        trg_tensor = torch.tensor(trg_batch).to(self.device)

        # 生成decoder输入输出
        trg_input = trg_tensor[:, :-1]  # 移除最后一个token
        trg_output = trg_tensor[:, 1:]  # 移除第一个token

        # 生成mask
        src_pad_mask = (src_tensor == 0).to(self.device)
        trg_pad_mask = (trg_input == 0).to(self.device)

        return {
            'src': src_tensor,
            'trg_input': trg_input,
            'trg_output': trg_output,
            'src_pad_mask': src_pad_mask,
            'trg_pad_mask': trg_pad_mask
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

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'config': self.config
        }

        filename = f"checkpoint_epoch_{epoch}.pt" if not is_best else "best_model.pt"
        torch.save(state, os.path.join(self.config['output_dir'], filename))
        logging.info(f"Checkpoint saved: {filename}")

    def train(self, train_loader, val_loader=None):
        best_loss = float('inf')

        for epoch in range(1, self.config['num_epochs'] + 1):
            logging.info(f"Epoch {epoch}/{self.config['num_epochs']}")

            train_loss = self.train_epoch(train_loader)
            logging.info(f"Train Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self.validate(val_loader)
                logging.info(f"Val Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)

            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)


if __name__ == "__main__":
    # 加载训练配置
    with open('./config.yml') as f:
        config = yaml.safe_load(f)['train']

    # 初始化训练器
    trainer = TransformerTrainer(config)

    try:
        trainer.train(train_loader, valid_loader)
    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint(epoch='interrupted')
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
