'''
import, and get:

- src_vocab, trg_vocab
- train_loader, valid_loader, test_loader
'''

import os
import yaml
import torch
from download import download
from torch.utils.data import Dataset, DataLoader

from vocab import Vocab


# 定义自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, max_length=50):
        """
        初始化翻译数据集
        :param src_sentences: 源语言句子列表
        :param trg_sentences: 目标语言句子列表
        :param src_vocab: 源语言词表 (Vocab 对象)
        :param trg_vocab: 目标语言词表 (Vocab 对象)
        :param max_length: 最大序列长度
        """
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]

        # 编码源语言和目标语言句子
        src_encoded = self.src_vocab.encode(src_sentence, add_special_tokens=True, max_length=self.max_length)
        trg_encoded = self.trg_vocab.encode(trg_sentence, add_special_tokens=True, max_length=self.max_length)

        return src_encoded, trg_encoded  # 返回元组 (src, trg)


# 数据加载函数
def load_datasets(
    train_path, valid_path, test_path, src_vocab, trg_vocab, task, max_length=50, batch_size=32, drop_last=False
):
    """
    加载训练集、验证集和测试集
    :param train_path: 训练集路径
    :param valid_path: 验证集路径
    :param test_path: 测试集路径
    :param src_vocab: 源语言词表 (Vocab 对象)
    :param trg_vocab: 目标语言词表 (Vocab 对象)
    :param task: 任务，en->de or de->en
    :param max_length: 最大序列长度
    :param batch_size: 批量大小
    :param drop_last: 是否丢弃最后一个不完整的批次
    :return: 训练集、验证集和测试集的 DataLoader
    """
    # 加载句子
    train_src = Vocab.load_data(os.path.join(train_path, "train.de"))
    train_trg = Vocab.load_data(os.path.join(train_path, "train.en"))

    valid_src = Vocab.load_data(os.path.join(valid_path, "val.de"))
    valid_trg = Vocab.load_data(os.path.join(valid_path, "val.en"))

    test_src = Vocab.load_data(os.path.join(test_path, "test2016.de"))
    test_trg = Vocab.load_data(os.path.join(test_path, "test2016.en"))

    if task != 'de->en':
        train_src, train_trg = train_trg, train_src
        valid_src, valid_trg = valid_trg, valid_src
        test_src, test_trg, test_trg, test_src

    # 创建数据集
    train_dataset = TranslationDataset(train_src, train_trg, src_vocab, trg_vocab, max_length=max_length)
    valid_dataset = TranslationDataset(valid_src, valid_trg, src_vocab, trg_vocab, max_length=max_length)
    test_dataset = TranslationDataset(test_src, test_trg, src_vocab, trg_vocab, max_length=max_length)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: list(zip(*batch)),
        drop_last=drop_last,  # 控制是否丢弃最后一个批次
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: list(zip(*batch)),
        drop_last=drop_last,  # 控制是否丢弃最后一个批次
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: list(zip(*batch)),
        drop_last=drop_last,  # 控制是否丢弃最后一个批次
    )

    return train_loader, valid_loader, test_loader


'''
1. 数据集下载与解压
2. 构建德文和英文的词表
3. 构建DataLoader
'''
url = "https://modelscope.cn/api/v1/datasets/SelinaRR/Multi30K/repo?Revision=master&FilePath=Multi30K.zip"
datasets_path = "./datasets/"
train_path = os.path.join(datasets_path, "train/")
valid_path = os.path.join(datasets_path, "valid/")
test_path = os.path.join(datasets_path, "test/")

if not os.path.exists(datasets_path):
    download(url, "./", kind="zip", replace=True)
    print("Dataset downloaded and extracted.")
else:
    print("Dataset is already downloaded.")


with open('./config.yml') as f:
    config = yaml.safe_load(f)

    task = config['data']['task']
    max_length = config['data']['max_length']
    batch_size = config['data']['batch_size']


de_sentences = Vocab.load_data(os.path.join(train_path, "train.de"))
en_sentences = Vocab.load_data(os.path.join(train_path, "train.en"))

de_vocab = Vocab()
de_vocab.build_vocab(de_sentences)

en_vocab = Vocab()
en_vocab.build_vocab(en_sentences)

print("德文词表大小:", len(de_vocab.word2idx))
print("英文词表大小:", len(en_vocab.word2idx))


# 词表
src_vocab, trg_vocab = (de_vocab, en_vocab) if task == 'de->en' else (en_vocab, de_vocab)


train_loader, valid_loader, test_loader = load_datasets(
    train_path=train_path,
    valid_path=valid_path,
    test_path=test_path,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    task=task,
    max_length=max_length,
    batch_size=batch_size,
    drop_last=True
)


if __name__ == '__main__':
    # 测试 DataLoader
    for batch in train_loader:
        src_batch, trg_batch = batch  # 解包元组
        src_batch = [torch.tensor(seq) for seq in src_batch]  # 转换为张量
        trg_batch = [torch.tensor(seq) for seq in trg_batch]  # 转换为张量

        src_batch = torch.stack(src_batch)  # 堆叠成一个批次
        trg_batch = torch.stack(trg_batch)  # 堆叠成一个批次

        print("Batch Source Shape:", src_batch.shape)  # [batch_size, max_length]
        print("Batch Target Shape:", trg_batch.shape)  # [batch_size, max_length]
        break  # 只打印一个批次以测试

    # 测试单个句子的编码和解码
    encoded_src = src_batch[1].tolist()
    encoded_trg = trg_batch[1].tolist()

    print(f'len src: {len(encoded_src)}, len trg: {len(encoded_trg)}')

    decoded_src = src_vocab.decode(encoded_src)
    decoded_trg = trg_vocab.decode(encoded_trg)

    print("\nTest Encoding and Decoding:")
    print("Encoded src Sentence :", encoded_src)
    print("Decoded src Sentence :", decoded_src)
    print()
    print("Encoded trg Sentence :", encoded_trg)
    print("Decoded trg Sentence :", decoded_trg)
