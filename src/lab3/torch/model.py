import torch
from torch import nn


'''
B: batch_size
L: max_len
H: hidden size of K, Q, V after mapping from input
D: embedding_size
'''


# 旋转位置编码
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size, p=0., max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p)

        pe = torch.zeros(max_len, embedding_size)  # (L, D)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,) -> (L, 1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / embedding_size))  # (D//2, )

        # (L, 1) * (D/2, ) -> (L, D/2) * 2 -> (L, D//2)
        pe[:, 0::2] = torch.sin(position * div_term)  # col列号, row是行号：pe[row] = sin(i / 10000^{col/D})
        pe[:, 1::2] = torch.cos(position * div_term)  # col奇数, row是行号：pe[row] = cos(i / 10000^{col-1/D})

        pe = pe.unsqueeze(0)  # (L, D) -> (1, L, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: (B, L, D) -> x + pos
        '''

        _, seq_len, _ = x.size()

        pos_encoding = self.pe[:, :seq_len, :]  # (1, L, D)
        x = x + pos_encoding  # (B, L, D)
        return self.dropout(x)


# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, p=0.):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v, mask=None):
        '''
        q: (B, [N], Lq, D)
        k: (B, [N], Lk, D)
        v: (B, [N], Lk, D)
        mask: (B, [N], Lq, Hk), fill: -inf

        -> res: (B, [N], Lq, D), atten: (B, [N], Lk, Lk)
        '''
        embedding_size = q.size(dim=0)
        d = torch.sqrt(torch.tensor(embedding_size, dtype=torch.float16))

        attention = torch.matmul(q, k.transpose(-1, -2) / d)  # (B, [N], Lq, D) @ (B, [N], D, Lk) / (1,) -> (B, [N], Lq, Lk)

        if mask is not None:
            attention = attention.masked_fill(mask, -torch.inf)  # mask: (B, 1, Lq, Lk) -> (B, [N], Lq, Lk), set True -> -inf

        attention = self.softmax(attention)  # softmax on dim of Hk
        attention = self.dropout(attention)

        res = torch.matmul(attention, v)  # (B, [N], Lq, Lk) @ (B, [N], Lk, D) -> (B, [N], Lq, D), 这里就是对Lk所在的维度进行加权平均

        return res, attention


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, new_embedding_size, head_size, p=0.):
        super(MultiHeadAttention, self).__init__()

        self.n_head = head_size
        self.new_embedding_size = new_embedding_size
        self.linear_q = nn.Linear(embedding_size, self.new_embedding_size * self.n_head)
        self.linear_k = nn.Linear(embedding_size, self.new_embedding_size * self.n_head)
        self.linear_v = nn.Linear(embedding_size, self.new_embedding_size * self.n_head)
        self.linear_o = nn.Linear(self.new_embedding_size * self.n_head, embedding_size)
        self.attention = SelfAttention(p=p)

    def forward(self, q: torch.Tensor, k, v, mask: torch.Tensor):
        '''
        q: (B, Lq, D)
        k: (B, Lk, D)
        v: (B, Lk, D)
        mask: (Lq, Lk), fill: -inf

        -> res: (B, Lq, D), atten: (B, N, Lq, Lk)
        '''
        batch_size = q.size(dim=0)

        Q = self.linear_q(q).view(batch_size, -1, self.n_head, self.new_embedding_size).transpose(1, 2)  # (B, Lq, N * D') -> (B, Lq, N, D') -> (B, N, Lq, D')
        K = self.linear_k(k).view(batch_size, -1, self.n_head, self.new_embedding_size).transpose(1, 2)  # (B, Lk, N * D') -> (B, Lk, N, D') -> (B, N, Lk, D')
        V = self.linear_v(v).view(batch_size, -1, self.n_head, self.new_embedding_size).transpose(1, 2)  # (B, Lk, N * D') -> (B, Lk, N, D') -> (B, N, Lk, D')

        mask = mask.unsqueeze(1)  # (B, 1, Lq, Lk)
        mask = mask.expand((-1, self.n_head, -1, -1))  # (B, N, Lq, Lk)

        res, atten = self.attention(Q, K, V, mask)  # (B, N, Lq, D')

        res = res.transpose(1, 2).reshape(batch_size, -1, self.new_embedding_size * self.n_head)  # (B, N, Lq, D') -> (B, Lq, N, D') -> (B, Lq, N * D')
        res = self.linear_o(res)  # (B, Lq, N * D') -> (B, Lq, D)

        return res, atten


# 位置前馈神经网络
class FFN(nn.Module):
    def __init__(self, hidden_size, embedding_size, p=0.):
        super(FFN, self).__init__()

        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        res = self.linear1(x)
        res = nn.ReLU(res)
        res = self.dropout(res)
        res = self.linear2(res)
        return res


# Add & Norm
class AddNorm(nn.Module):
    def __init__(self, embedding_size, p=0.):
        super(AddNorm, self).__init__()

        self.LN = nn.LayerNorm((embedding_size,))
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, fx):
        fx = self.dropout(fx)
        return self.LN(x + fx)


# Mask = <PAD> + SeeForward
class Mask:
    '''
    where "True" is the place should be masked.

    seq_q: (B, Lq), seq after vocab encode for q.
    seq_k: (B, Lk), seq after vocab encode for k.

    -> (B, Lq, Lk), mask
    '''

    def get_padding_mask(self, seq_q, seq_k, who_is_pad=0):
        '''
        <PAD> mask

        seq_q: (B, Lq), seq after vocab encode for q.
        seq_k: (B, Lk), seq after vocab encode for k.

        -> (B, Lq, Lk), mask
        '''

        batch_size, Lq = seq_q.size()
        batch_size, Lk = seq_k.size()

        pad_mask = (seq_k == who_is_pad)  # (B, Lk)
        pad_mask = pad_mask.unsqueeze(1).expand(batch_size, Lq, Lk)  # (B, Lk) -> (B, 1, Lk) -> (B, Lq, Lk)

        return pad_mask

    def get_causal_mask(self, seq_q, seq_k):
        '''
        causal mask

        seq_q: (B, Lq), seq after vocab encode for q.
        seq_k: (B, Lk), seq after vocab encode for k.

        -> (B, Lq, Lk), mask
        '''

        B, Lq = seq_q.size()
        _, Lk = seq_k.size()

        mask = ~torch.tril(torch.ones(Lq, Lk)).bool()

        mask = mask.unsqueeze(0).expand(B, -1, -1)

        return mask


'''
Encoder
'''


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, head_size, ffn_size, p=0.):
        super(EncoderBlock, self).__init__()

        new_embedding_size = embedding_size // head_size

        if new_embedding_size * head_size != embedding_size:
            raise ValueError(f'make sure embedding_size % head_size == 0, but get: {embedding_size} and {head_size}.')

        self.encoder_self_attention = MultiHeadAttention(embedding_size, new_embedding_size, head_size, p)
        self.ffn = FFN(ffn_size, p)

        self.AN1 = AddNorm(embedding_size, p)
        self.AN2 = AddNorm(embedding_size, p)

    def forward(self, encoder_input, encoder_self_mask):
        # fx branch
        x = encoder_input

        fx, _ = self.encoder_self_attention(q=x, k=x, v=x, mask=encoder_self_mask)
        out1 = self.AN1(x, fx)

        x = out1

        fx = self.ffn(x)
        res = self.AN2(x, fx)

        return res


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, head_size, ffn_size, num_blocks, p=0.):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding(src_vocab_size, embedding_size)
        self.pos_embed = PositionalEncoding(embedding_size, p)

        self.blocks = [
            EncoderBlock(
                embedding_size=embedding_size,
                head_size=head_size,
                ffn_size=ffn_size,
                p=p
            ) for _ in range(num_blocks)
        ]

        self.scaling = torch.sqrt(torch.tensor(embedding_size))

    def forward(self, encoder_input, src_who_is_pad):
        embeded_encoder_input = self.embed(encoder_input)
        embeded_encoder_input = self.pos_embed(embeded_encoder_input)

        encoder_self_mask = Mask().get_padding_mask(seq_q=encoder_input,
                                                    seq_k=encoder_input,
                                                    who_is_pad=src_who_is_pad)

        encoder_output = embeded_encoder_input
        for block in self.blocks:
            encoder_output = block(encoder_output, encoder_self_mask)

        return encoder_output


'''
Decoder
'''


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, head_size, ffn_size, p=0.):
        super(DecoderBlock, self).__init__()

        one_head_embedding_size = embedding_size // head_size
        if one_head_embedding_size * head_size != embedding_size:
            raise ValueError(f'make sure embedding_size % head_size == 0, but get: {embedding_size} and {head_size}.')

        self.decoder_self_attention = MultiHeadAttention(embedding_size, one_head_embedding_size, head_size, p)
        self.decoder_encoder_attention = MultiHeadAttention(embedding_size, one_head_embedding_size, head_size, p)

        self.ffn = FFN(ffn_size, embedding_size, p)

        self.AN1 = AddNorm(embedding_size, p)
        self.AN2 = AddNorm(embedding_size, p)
        self.AN3 = AddNorm(embedding_size, p)

    def forward(self, decoder_input, encoder_output, decoder_self_mask, decoder_encoder_mask):
        x = decoder_input

        fx, _ = self.decoder_self_attention(q=x, k=x, v=x, mask=decoder_self_mask)
        out1 = self.AN1(x, fx)

        x = out1
        fx, _ = self.decoder_encoder_attention(q=x, k=encoder_output, v=encoder_output, mask=decoder_self_mask)
        out2 = self.AN2(x, fx)

        x = out2

        fx = self.ffn(x)
        res = self.AN3(x, fx)

        return res


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_size, head_size, ffn_size, num_blocks, p=0.):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(trg_vocab_size, embedding_size)
        self.pos_embed = PositionalEncoding(embedding_size, p)

        self.blocks = [
            DecoderBlock(
                embedding_size=embedding_size,
                head_size=head_size,
                ffn_size=ffn_size,
                p=p
            ) for _ in range(num_blocks)
        ]

        self.scaling = torch.sqrt(torch.tensor(embedding_size))
        self.linear_out = nn.Linear(embedding_size, trg_vocab_size)

    def forward(self, decoder_input, encoder_input, encoder_output, src_who_is_pad, trg_who_is_pad):
        embeded_decoder_input = self.embed(decoder_input)
        embeded_decoder_input = self.pos_embed(embeded_decoder_input)

        decoder_self_padding_mask = Mask().get_padding_mask(decoder_input, decoder_input, trg_who_is_pad)
        decoder_self_causal_mask = Mask().get_causal_mask(decoder_input, decoder_input)
        decoder_self_mask = decoder_self_padding_mask and decoder_self_causal_mask

        decoder_encoder_padding_mask = Mask().get_padding_mask(decoder_input, encoder_input, src_who_is_pad)
        decoder_encoder_mask = decoder_encoder_padding_mask

        decoder_output = embeded_decoder_input
        for block in self.blocks:
            decoder_output = block(decoder_output, encoder_output, decoder_self_mask, decoder_encoder_mask)

        decoder_output = self.linear_out(decoder_output)

        return decoder_output


'''
Transformer
'''


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, src_who_is_pad, trg_who_is_pad):
        encoder_output = self.encoder(encoder_input, src_who_is_pad)

        decoder_output = self.decoder(decoder_input, encoder_input, encoder_output, src_who_is_pad, trg_who_is_pad)

        logits = decoder_output.view((-1, decoder_output.shape[-1]))

        return logits


# 测试代码
if __name__ == '__main__':
    def test_positional_encoding():

        # 定义模型
        class TestModel(nn.Module):
            def __init__(self, vocab_size, d_model, dropout_p=0.1, max_len=100):
                super(TestModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)  # 随机初始化词嵌入
                self.positional_encoding = PositionalEncoding(d_model, dropout_p, max_len)

            def forward(self, src):
                """
                :param src: 输入序列，形状为 (batch_size, seq_len)
                :return: 添加了位置编码的嵌入表示
                """
                # 通过 Embedding 层
                embedded = self.embedding(src)  # (batch_size, seq_len, d_model)
                print("\nEmbedding 输出:")
                print(embedded)
                print("形状:", embedded.shape)

                # 添加位置编码
                output = self.positional_encoding(embedded)  # (batch_size, seq_len, d_model)
                return output

        # 假设词汇表大小和嵌入维度
        vocab_size = 10000  # 假设词汇表大小为 10000
        d_model = 4         # 嵌入维度
        batch_size = 4      # 批次大小
        seq_len = 10        # 序列长度

        # 创建模型
        model = TestModel(vocab_size=vocab_size, d_model=d_model)

        # 创建模拟数据
        src = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机生成输入序列
        print("原始输入序列:")
        print(src)

        # 前向传播
        output = model(src)
        print("\n最终输出形状:")
        print(output.shape)  # 输出形状应为 (batch_size, seq_len, d_model)

        # 打印部分输出值
        print("\n最终输出的部分值:")
        print(output[0, :, :10])  # 打印第一个样本的前 10 维特征

    def test_attention():
        # 参数设置
        batch_size = 2
        seq_len_q = 3
        seq_len_k = 4
        seq_len_v = 4
        embedding_size = 5
        dropout_p = 0.

        # 随机生成输入张量
        q = torch.randn(batch_size, seq_len_q, embedding_size)  # 查询张量
        k = torch.randn(batch_size, seq_len_k, embedding_size)  # 键张量
        v = torch.randn(batch_size, seq_len_v, embedding_size)  # 值张量

        # 创建掩码（可选）
        mask = torch.zeros(batch_size, seq_len_q, seq_len_k, dtype=torch.bool)
        mask[:, :, 2:] = True  # 掩码掉部分位置

        # 初始化自注意力机制
        self_attention = SelfAttention(p=dropout_p)

        # 前向传播
        output, attention_weights = self_attention(q, k, v, mask=mask)

        # 打印结果
        print("查询张量 q:")
        print(q)
        print("\n键张量 k:")
        print(k)
        print("\n值张量 v:")
        print(v)
        print("\n掩码 mask:")
        print(mask)
        print("\n输出张量 output:")
        print(output)
        print("\n注意力权重 attention_weights:")
        print(attention_weights)

        # 检查输出形状
        assert output.shape == (batch_size, seq_len_q, embedding_size), f"输出形状错误: {output.shape}"
        assert attention_weights.shape == (batch_size, seq_len_q, seq_len_k), f"注意力权重形状错误: {attention_weights.shape}"

        print("\n测试通过！")

    def test_multi_head_attention():
        # 参数设置
        batch_size = 128
        seq_len_q = 5
        seq_len_k = 6
        embedding_size = 8
        new_embedding_size = 4
        head_size = 4
        dropout_p = 0.

        # 随机生成输入张量
        q = torch.randn(batch_size, seq_len_q, embedding_size)  # 查询张量 (B, Hq, D)
        k = torch.randn(batch_size, seq_len_k, embedding_size)  # 键张量 (B, Hk, D)
        v = torch.randn(batch_size, seq_len_k, embedding_size)  # 值张量 (B, Hk, D)

        # 创建掩码（可选）
        mask = torch.zeros(batch_size, seq_len_q, seq_len_k, dtype=torch.bool)  # (B, Hq, Hk)
        mask[:, :, 4:] = True  # 掩码掉部分位置

        # 初始化多头注意力机制
        multi_head_attention = MultiHeadAttention(
            embedding_size=embedding_size,
            new_embedding_size=new_embedding_size,
            head_size=head_size,
            p=dropout_p
        )

        # 前向传播
        output, attention_weights = multi_head_attention(q, k, v, mask)

        # 打印结果
        print("查询张量 q:")
        print(q.size())
        print("\n键张量 k:")
        print(k.size())
        print("\n值张量 v:")
        print(v.size())
        print("\n掩码 mask:")
        print(mask.size())
        print("\n输出张量 output:")
        print(output.size())
        print("\n注意力权重 attention_weights:")
        print(attention_weights.size())

        # 检查输出形状
        assert output.shape == (batch_size, seq_len_q, embedding_size), f"输出形状错误: {output.shape}"
        assert attention_weights.shape == (batch_size, head_size, seq_len_q, seq_len_k), f"注意力权重形状错误: {attention_weights.shape}"

        print("\n测试通过！")

    # test_positional_encoding()
    # test_attention()
    test_multi_head_attention()
