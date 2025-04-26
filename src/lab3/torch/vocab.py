import re
from collections import Counter


class Vocab:
    def __init__(self, special_tokens=None):
        """
        :param special_tokens: 特殊标记列表，默认包含 '<PAD>': 0, '<UNK>':1, '<BOS>':2, '<EOS>':3
        """
        self.word2idx = {}
        self.idx2word = {}
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for token in special_tokens:
            self.add_word(token)

    def add_word(self, word):
        """
        向词表中添加单词
        :param word: 要添加的单词
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_vocab(self, sentences, min_freq=1):
        """
        基于句子列表构建词表，根据词频排序分配索引
        :param sentences: 句子列表
        :param min_freq: 最小词频限制
        """
        # 统计词频
        word_freq = Counter()
        for sentence in sentences:
            words = self.tokenize(sentence)  # 使用自定义的分词函数
            for word in words:
                word_freq[word] += 1

        # 根据频率过滤词汇并排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 添加高频词到词表中
        for word, _ in sorted_words:
            if word_freq[word] >= min_freq:
                self.add_word(word)

    def encode(self, sentence, add_special_tokens=True, max_length=None):
        """
        将句子转换为索引序列，并可选择性地添加<BOS>和<EOS>
        :param sentence: 输入句子
        :param add_special_tokens: 是否添加特殊标记
        :param max_length: 句子的最大长度是多少，少了在末尾加入<PAD>，否则截断。
        :return: 索引序列
        """
        tokens = self.tokenize(sentence)
        if add_special_tokens:
            tokens = ["<BOS>"] + tokens + ["<EOS>"]

        arr = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]

        if max_length is None:
            return arr
        else:
            if len(arr) < max_length:
                arr += [0] * (max_length - len(arr))
                return arr
            else:
                return arr[: max_length - 1] + [3]

    def decode(self, indices, ignore_special_tokens=False):
        """
        将索引序列转换回句子，并可选择性地忽略特殊标记
        :param indices: 索引序列
        :param ignore_special_tokens: 是否忽略特殊标记
        :return: 句子
        """
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, "<UNK>")
            if ignore_special_tokens and word in ["<PAD>", "<BOS>", "<EOS>"]:
                continue
            words.append(word)
        return words

    @staticmethod
    def load_data(file_path):
        """加载并返回文件中的所有句子"""
        with open(file_path, "r", encoding="utf-8") as file:
            sentences = file.readlines()
        return [sentence.strip() for sentence in sentences]

    def save_vocab(self, path):
        """将词表保存到指定路径"""
        with open(path, "w", encoding="utf-8") as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")

    def load_vocab(self, path):
        """从指定路径加载词表"""
        self.word2idx = {}
        self.idx2word = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word, idx = line.strip().split("\t")
                idx = int(idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokenize(self, sentence):
        """
        分词函数，保留标点符号作为独立的标记
        :param sentence: 输入句子
        :return: 分词后的列表
        """
        # 使用正则表达式分离单词和标点符号
        words_and_punct = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        return words_and_punct


if __name__ == "__main__":
    from data_process import train_path

    # 加载德文和英文训练数据
    de_sentences = Vocab.load_data(train_path + "train.de")
    en_sentences = Vocab.load_data(train_path + "train.en")

    # 构建德文和英文的词表
    de_vocab = Vocab()
    de_vocab.build_vocab(de_sentences)

    en_vocab = Vocab()
    en_vocab.build_vocab(en_sentences)

    # 保存词表
    # de_vocab.save_vocab('./vocab_de.txt')
    # en_vocab.save_vocab('./vocab_en.txt')

    print("德文词表大小:", len(de_vocab.word2idx))
    print("英文词表大小:", len(en_vocab.word2idx))

    # 示例1：德语
    print(">>> 德语")
    sample_sentence = "Ein Mann hält eine Kamera."
    print("原始的句子  :", sample_sentence)
    encoded = de_vocab.encode(sample_sentence, max_length=15)
    print("编码后的句子:", encoded)
    decoded = de_vocab.decode(encoded)
    print("解码后的句子:", decoded)

    # 示例2：英语小写
    print(">>> 英语小写")
    sample_sentence = (
        "I'm your father, I'm your father, I'm your father, I'm your father."
    )
    print("原始的句子  :", sample_sentence)
    encoded = en_vocab.encode(sample_sentence, max_length=5)
    print("编码后的句子:", encoded)
    decoded = en_vocab.decode(encoded)
    print("解码后的句子:", decoded)

    # 示例2：英语大写
    print(">>> 英语大写")
    sample_sentence = "I'm yOuR Father."
    print("原始的句子  :", sample_sentence)
    encoded = en_vocab.encode(sample_sentence, max_length=15)
    print("编码后的句子:", encoded)
    decoded = en_vocab.decode(encoded)
    print("解码后的句子:", decoded)
