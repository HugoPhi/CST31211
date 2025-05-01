import torch
from torch import nn


'''
B: batch_size
L: max_len
H: hidden size of K, Q, V after mapping from input
D: embedding_size
'''
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # 检查 Apple Silicon GPU 是否可用
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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
        embedding_size = q.size(dim=-1)
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
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        x: (B, L, D)

        -> (B, L, D)
        '''

        res = self.linear1(x)
        res = self.relu(res)
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
        '''
        x: (B, L, D)

        -> (B, L, D)
        '''

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
        self.ffn = FFN(ffn_size, embedding_size, p)

        self.AN1 = AddNorm(embedding_size, p)
        self.AN2 = AddNorm(embedding_size, p)

    def forward(self, encoder_input, encoder_self_mask):
        '''
        encoder_input: (B, L) ~int ~raw_sentence
        encoder_self_mask: (B, L, L) ~bool

        -> (B, L, D) ~float
        '''

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

        self.blocks = nn.ModuleList([
            EncoderBlock(
                embedding_size=embedding_size,
                head_size=head_size,
                ffn_size=ffn_size,
                p=p
            ) for _ in range(num_blocks)
        ])

        self.scaling = torch.sqrt(torch.tensor(embedding_size))

    def forward(self, encoder_input, src_who_is_pad):
        '''
        encoder_input: (B, L) ~int ~raw_sentence
        src_who_is_pad: (,) ~int

        -> (B, L, D) ~float
        '''

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
        '''
        decoder_input: (B, L) ~int ~raw_sentence
        encoder_output: (B, L, D) ~float
        decoder_self_mask: (B, L, L) ~bool
        decoder_encoder_mask: (B, L, L) ~bool

        -> (B, L, D) ~float
        '''

        x = decoder_input

        fx, _ = self.decoder_self_attention(q=x, k=x, v=x, mask=decoder_self_mask)
        out1 = self.AN1(x, fx)

        x = out1
        fx, _ = self.decoder_encoder_attention(q=x, k=encoder_output, v=encoder_output, mask=decoder_encoder_mask)
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

        self.blocks = nn.ModuleList([
            DecoderBlock(
                embedding_size=embedding_size,
                head_size=head_size,
                ffn_size=ffn_size,
                p=p
            ) for _ in range(num_blocks)
        ])

        self.scaling = torch.sqrt(torch.tensor(embedding_size))
        self.linear_out = nn.Linear(embedding_size, trg_vocab_size)

    def forward(self, decoder_input, encoder_input, encoder_output, src_who_is_pad, trg_who_is_pad):
        '''
        decoder_input: (B, L) ~int ~raw_sentence
        encoder_input: (B, L) ~int ~raw_sentence
        encoder_output: (B, L, D) ~float
        src_who_is_pad, trg_who_is_pad: (,) ~int

        -> (B, L, trg_vocab_size) ~float
        '''

        embeded_decoder_input = self.embed(decoder_input)
        embeded_decoder_input = self.pos_embed(embeded_decoder_input)

        decoder_self_padding_mask = Mask().get_padding_mask(decoder_input, decoder_input, trg_who_is_pad)
        decoder_self_causal_mask = Mask().get_causal_mask(decoder_input, decoder_input)
        decoder_self_mask = decoder_self_padding_mask.to(device) | decoder_self_causal_mask.to(device)  # can not use 'or' here

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
        '''
        encoder_input: (B, L) ~int ~raw_sentence
        decoder_input: (B, L) ~int ~raw_sentence
        src_who_is_pad, trg_who_is_pad: (,) ~int

        -> (B*L, trg_vocab_size)
        '''
        encoder_output = self.encoder(encoder_input, src_who_is_pad)

        decoder_output = self.decoder(decoder_input, encoder_input, encoder_output, src_who_is_pad, trg_who_is_pad)

        logits = decoder_output.view((-1, decoder_output.shape[-1]))

        return logits


# Tests by DeepSeek
# 1. Position Encoding with D=6
# 2. Model structure
# 3. Shape of out for Encoder, Decoder, Transformer
if __name__ == '__main__':
    import torchinfo
    device = torch.device("cpu")

    # 设置固定随机种子
    torch.manual_seed(42)
    
    # 1. 位置编码测试 (D=6)
    print("\n=== 位置编码输出 (D=6) ===")
    pe = PositionalEncoding(embedding_size=6, max_len=3)
    dummy_input = torch.zeros(1, 3, 6)  # (B, L, D)
    print("位置编码矩阵:")
    print(pe.pe.squeeze(0))  # 显示3个位置6维的编码
    
    # 初始化小模型
    encoder = Encoder(
        src_vocab_size=1000,
        embedding_size=512,
        head_size=4,
        ffn_size=1024,
        num_blocks=2
    )
    
    decoder = Decoder(
        trg_vocab_size=1500,
        embedding_size=512,
        head_size=4,
        ffn_size=1024,
        num_blocks=2
    )
    
    transformer = Transformer(encoder, decoder)
    # 2. 打印结构
    print('>> Encoder:')
    torchinfo.summary(
        encoder,
        input_data=(
            torch.randint(low=1, high=100, size=(128, 15)),
            0
        ),
        col_names=["input_size", "output_size", "num_params", "trainable"]
    )

    print('\n>> Decoder:')
    torchinfo.summary(
        decoder,
        input_data=(
            torch.randint(low=1, high=100, size=(128, 15)),
            torch.randint(low=1, high=100, size=(128, 15)),
            torch.rand(128, 15, 512),
            0,
            0
        ),
        col_names=["input_size", "output_size", "num_params", "trainable"]
    )

    print('\n>> Transformer:')
    torchinfo.summary(
        transformer,
        input_data=(
            torch.randint(low=1, high=100, size=(128, 15)),
            torch.randint(low=1, high=100, size=(128, 15)),
            0,
            0
        ),
        col_names=["input_size", "output_size", "num_params", "trainable"]
    )

    # 3. 输入输出形状验证
    print("\n=== 形状验证 ===")
    # 测试数据
    src = torch.randint(0, 1000, (2000, 5))  # (B, L_src)
    trg = torch.randint(0, 1500, (2000, 7))  # (B, L_trg)

    enc_out = encoder(src, 0)
    print(f"Encoder输入: {src.shape} → 输出: {enc_out.shape}")

    dec_out = decoder(trg, src, enc_out, 0, 0)
    print(f"Decoder输入: {trg.shape} → 输出: {dec_out.shape}")

    final_out = transformer(src, trg, 0, 0)
    print(f"Transformer输入: src:{src.shape}_trg:{trg.shape} → 输出: {final_out.shape}")
