import torch
from torch import nn


'''
B: batch_size
L: max_len
H: hidden size of K, Q, V after mapping from input
D: embedding_size
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        '''
        encoder_output = self.encoder(encoder_input, src_who_is_pad)

        decoder_output = self.decoder(decoder_input, encoder_input, encoder_output, src_who_is_pad, trg_who_is_pad)

        logits = decoder_output.view((-1, decoder_output.shape[-1]))

        return logits


# 测试代码
if __name__ == '__main__':
    def test_positional_encoding_shapes():
        d_model = 64
        max_len = 100
        batch_size = 16
        seq_len = 50
        pe = PositionalEncoding(d_model, max_len=max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        assert output.shape == (batch_size, seq_len, d_model), "PositionalEncoding shape mismatch"

    def test_self_attention_shapes():
        batch_size = 4
        seq_len = 10
        d_model = 32
        sa = SelfAttention()
        q = torch.randn(batch_size, seq_len, d_model)
        k = v = torch.randn(batch_size, seq_len, d_model)
        output, attn = sa(q, k, v)
        assert output.shape == q.shape, "SelfAttention output shape mismatch"
        assert attn.shape == (batch_size, seq_len, seq_len), "SelfAttention attention shape mismatch"

    def test_multi_head_attention_shapes():
        d_model = 64
        n_head = 4
        new_d = d_model // n_head
        mha = MultiHeadAttention(d_model, new_d, n_head)
        batch_size = 8
        seq_len = 15
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        output, attn = mha(x, x, x, mask)
        assert output.shape == x.shape, "MHA output shape mismatch"
        assert attn.shape == (batch_size, n_head, seq_len, seq_len), "MHA attention shape mismatch"

    def test_encoder_block_shapes():
        d_model = 128
        head_size = 8
        ffn_size = 256
        batch_size = 16
        seq_len = 20
        eb = EncoderBlock(d_model, head_size, ffn_size)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        output = eb(x, mask)
        assert output.shape == x.shape, "EncoderBlock output shape mismatch"

    def test_encoder_decoder_data_flow():
        # 配置参数
        batch_size = 32
        src_seq_len = 50
        trg_seq_len = 50
        d_model = 512
        n_head = 8
        ffn_size = 2048
        num_blocks = 3
        src_vocab_size = 10000
        trg_vocab_size = 15000

        # 初始化模型
        encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embedding_size=d_model,
            head_size=n_head,
            ffn_size=ffn_size,
            num_blocks=num_blocks,
            p=0.1
        )

        decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embedding_size=d_model,
            head_size=n_head,
            ffn_size=ffn_size,
            num_blocks=num_blocks,
            p=0.1
        )

        # 生成测试数据 ------------------------------------------------------------
        # 原始输入（token索引）
        src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # (32, 50)
        trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))  # (32, 50)

        print("\n[输入验证]")
        print(f"源序列形状: {src.shape} (batch_size, src_seq_len)")
        print(f"目标序列形状: {trg.shape} (batch_size, trg_seq_len)")

        # 编码器数据流测试 --------------------------------------------------------
        print("\n[编码器阶段]")

        # 1. 词嵌入
        embed_layer = encoder.embed
        embed_output = embed_layer(src)  # (32, 50, 512)
        print(f"嵌入层输出形状: {embed_output.shape} (batch_size, seq_len, d_model)")
        assert embed_output.shape == (batch_size, src_seq_len, d_model)

        # 2. 位置编码
        pos_encoder = encoder.pos_embed
        pos_output = pos_encoder(embed_output)  # (32, 50, 512)
        print(f"位置编码输出形状: {pos_output.shape} (batch_size, seq_len, d_model)")
        assert pos_output.shape == (batch_size, src_seq_len, d_model)

        # 3. 编码器块处理
        encoder_mask = Mask().get_padding_mask(src, src, 0)  # (32, 50, 50)
        encoder_output = pos_output
        for i, block in enumerate(encoder.blocks):
            encoder_output = block(encoder_output, encoder_mask)
            print(f"编码器块 {i + 1} 输出形状: {encoder_output.shape}")
            assert encoder_output.shape == (batch_size, src_seq_len, d_model)

        # 解码器数据流测试 --------------------------------------------------------
        print("\n[解码器阶段]")

        # 1. 词嵌入
        trg_embed_layer = decoder.embed
        trg_embed_output = trg_embed_layer(trg)  # (32, 50, 512)
        print(f"目标嵌入层输出形状: {trg_embed_output.shape}")
        assert trg_embed_output.shape == (batch_size, trg_seq_len, d_model)

        # 2. 位置编码
        trg_pos_output = decoder.pos_embed(trg_embed_output)  # (32, 50, 512)
        print(f"目标位置编码输出形状: {trg_pos_output.shape}")
        assert trg_pos_output.shape == (batch_size, trg_seq_len, d_model)

        # 3. 解码器块处理
        decoder_self_mask = (
            Mask().get_padding_mask(trg, trg, 0) | Mask().get_causal_mask(trg, trg)
        )
        decoder_encoder_mask = Mask().get_padding_mask(trg, src, 0)

        decoder_output = trg_pos_output
        for i, block in enumerate(decoder.blocks):
            decoder_output = block(
                decoder_output,
                encoder_output,
                decoder_self_mask,
                decoder_encoder_mask
            )
            print(f"解码器块 {i + 1} 输出形状: {decoder_output.shape}")
            assert decoder_output.shape == (batch_size, trg_seq_len, d_model)

        # 最终输出验证 ----------------------------------------------------------
        print("\n[最终输出]")
        logits = decoder.linear_out(decoder_output)  # (32, 50, 15000)
        logits = logits.view(-1, trg_vocab_size)     # (32*50, 15000)

        print("解码器最终输出形状:", logits.shape)
        assert logits.shape == (batch_size * trg_seq_len, trg_vocab_size), \
            f"形状错误: 期望 {(batch_size * trg_seq_len, trg_vocab_size)}, 实际 {logits.shape}"

        print("\n全流程形状验证通过！")

    def test_transformer_full_pipeline():
        # 配置参数
        batch_size = 32
        src_seq_len = 50
        trg_seq_len = 50
        d_model = 512
        n_head = 8
        ffn_size = 2048
        num_blocks = 6
        src_vocab_size = 10000
        trg_vocab_size = 15000

        # 初始化完整Transformer模型
        transformer = Transformer(
            encoder=Encoder(
                src_vocab_size=src_vocab_size,
                embedding_size=d_model,
                head_size=n_head,
                ffn_size=ffn_size,
                num_blocks=num_blocks,
                p=0.1
            ),
            decoder=Decoder(
                trg_vocab_size=trg_vocab_size,
                embedding_size=d_model,
                head_size=n_head,
                ffn_size=ffn_size,
                num_blocks=num_blocks,
                p=0.1
            )
        )

        # 生成测试数据 ------------------------------------------------------------
        src = torch.randint(1, src_vocab_size - 1, (batch_size, src_seq_len))  # 避免pad token
        trg = torch.randint(1, trg_vocab_size - 1, (batch_size, trg_seq_len))

        print("\n=== 输入验证 ===")
        print(f"源序列形状: {src.shape} (应满足: [32, 50])")
        print(f"目标序列形状: {trg.shape} (应满足: [32, 50])")

        # 完整前向传播流程验证 ----------------------------------------------------
        print("\n=== 编码器阶段验证 ===")

        # 1. 编码器嵌入层
        encoder_emb = transformer.encoder.embed(src)
        print(f"编码器嵌入输出形状: {encoder_emb.shape} (应满足: [32, 50, 512])")
        assert encoder_emb.requires_grad, "编码器嵌入应启用梯度"

        # 2. 位置编码验证
        pos_encoded = transformer.encoder.pos_embed(encoder_emb)
        print(f"位置编码输出形状: {pos_encoded.shape} (应满足: [32, 50, 512])")
        assert not torch.allclose(encoder_emb, pos_encoded), "位置编码应有变化"

        # 3. 编码器块处理
        encoder_mask = Mask().get_padding_mask(src, src, 0)
        encoder_output = pos_encoded
        for i in range(num_blocks):
            prev_output = encoder_output
            encoder_output = transformer.encoder.blocks[i](encoder_output, encoder_mask)
            print(f"编码器块 {i + 1} 输出形状: {encoder_output.shape}")
            assert encoder_output.shape == (batch_size, src_seq_len, d_model)
            assert not torch.allclose(prev_output, encoder_output), f"编码器块 {i + 1} 应有变化"

        print("\n=== 解码器阶段验证 ===")

        # 1. 解码器嵌入层
        decoder_emb = transformer.decoder.embed(trg)
        print(f"解码器嵌入输出形状: {decoder_emb.shape} (应满足: [32, 50, 512])")
        assert decoder_emb.requires_grad, "解码器嵌入应启用梯度"

        # 2. 解码器位置编码
        decoder_pos = transformer.decoder.pos_embed(decoder_emb)
        print(f"解码器位置编码输出形状: {decoder_pos.shape} (应满足: [32, 50, 512])")
        assert not torch.allclose(decoder_emb, decoder_pos), "位置编码应有变化"

        # 3. 解码器块处理
        decoder_self_mask = (
            Mask().get_padding_mask(trg, trg, 0) | Mask().get_causal_mask(trg, trg)
        )
        decoder_encoder_mask = Mask().get_padding_mask(trg, src, 0)

        decoder_output = decoder_pos
        for i in range(num_blocks):
            prev_output = decoder_output
            decoder_output = transformer.decoder.blocks[i](
                decoder_output,
                encoder_output,
                decoder_self_mask,
                decoder_encoder_mask
            )
            print(f"解码器块 {i + 1} 输出形状: {decoder_output.shape}")
            assert decoder_output.shape == (batch_size, trg_seq_len, d_model)
            assert not torch.allclose(prev_output, decoder_output), f"解码器块 {i + 1} 应有变化"

        # 最终输出验证 ----------------------------------------------------------
        print("\n=== 最终输出验证 ===")
        logits = transformer.decoder.linear_out(decoder_output)  # (32, 50, 15000)
        final_output = logits.view(-1, trg_vocab_size)           # (1600, 15000)

        print("最终输出形状:", final_output.shape)
        assert final_output.shape == (batch_size * trg_seq_len, trg_vocab_size)
        assert not torch.all(torch.isnan(final_output)), "输出包含NaN值"
        assert not torch.all(torch.isinf(final_output)), "输出包含Inf值"

        # 概率分布验证
        prob = torch.softmax(final_output, dim=-1)
        assert torch.allclose(prob.sum(dim=1), torch.ones(batch_size * trg_seq_len),
                              rtol=1e-3), "概率和不等于1"

        print("\n=== 反向传播验证 ===")
        # 模拟损失计算
        dummy_target = torch.randint(0, trg_vocab_size, (batch_size * trg_seq_len,))
        loss = torch.nn.functional.cross_entropy(final_output, dummy_target)
        loss.backward()

        # 检查关键参数梯度
        for name, param in transformer.named_parameters():
            assert param.grad is not None, f"参数 {name} 无梯度"
            assert not torch.all(param.grad == 0), f"参数 {name} 梯度全零"

        print("梯度流动验证通过")

    def test_transformer_end_to_end():
        # 配置参数
        batch_size = 4
        src_seq_len = 50
        trg_seq_len = 45
        d_model = 512
        n_head = 8
        ffn_size = 2048
        num_blocks = 3
        src_vocab_size = 10000
        trg_vocab_size = 15000

        # 初始化完整Transformer
        transformer = Transformer(
            encoder=Encoder(
                src_vocab_size=src_vocab_size,
                embedding_size=d_model,
                head_size=n_head,
                ffn_size=ffn_size,
                num_blocks=num_blocks,
                p=0.1
            ),
            decoder=Decoder(
                trg_vocab_size=trg_vocab_size,
                embedding_size=d_model,
                head_size=n_head,
                ffn_size=ffn_size,
                num_blocks=num_blocks,
                p=0.1
            )
        )

        # 生成测试数据 ------------------------------------------------------------
        # 创建含padding的真实模拟数据（假设pad_id=0）
        src = torch.randint(1, src_vocab_size - 1, (batch_size, src_seq_len))
        trg = torch.randint(1, trg_vocab_size - 1, (batch_size, trg_seq_len))

        # 添加随机padding（约20%的位置为0）
        src_mask = torch.rand(src.shape) < 0.2
        trg_mask = torch.rand(trg.shape) < 0.2
        src = src.masked_fill(src_mask, 0)
        trg = trg.masked_fill(trg_mask, 0)

        print("\n=== 输入验证 ===")
        print(f"源序列形状: {src.shape} (含 {src.eq(0).sum()} 个pad)")
        print(f"目标序列形状: {trg.shape} (含 {trg.eq(0).sum()} 个pad)")

        # 完整前向传播验证 --------------------------------------------------------
        print("\n=== 前向传播验证 ===")
        logits = transformer(
            encoder_input=src,
            decoder_input=trg,
            src_who_is_pad=0,
            trg_who_is_pad=0
        )

        # 验证输出形状
        expected_shape = (batch_size * trg_seq_len, trg_vocab_size)
        print(f"输出形状: {logits.shape} (应满足: {expected_shape})")
        assert logits.shape == expected_shape, "输出形状错误"

        # 验证数值合理性
        assert not torch.isnan(logits).any(), "输出包含NaN值"
        assert not torch.isinf(logits).any(), "输出包含Inf值"

        # 验证概率分布
        probs = torch.softmax(logits, dim=-1)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4), "概率和不等于1"

        # 注意力机制验证 ---------------------------------------------------------
        # 获取最后一个解码器块的注意力权重
        with torch.no_grad():
            encoder_output = transformer.encoder(src, 0)
            _ = transformer.decoder(trg, src, encoder_output, 0, 0)
            _ = transformer.decoder.blocks[-1]

        # 反向传播验证 ----------------------------------------------------------
        print("\n=== 反向传播验证 ===")
        transformer.zero_grad()

        # 模拟真实标签（忽略pad位置）
        labels = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_len))
        labels[trg == 0] = -100  # 忽略pad位置

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, trg_vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        loss.backward()

        # 检查梯度流动
        grad_exists = []
        for name, param in transformer.named_parameters():
            grad_valid = (param.grad is not None) and (param.grad.abs().sum() > 0)
            grad_exists.append(grad_valid)
            print(f"{name:50} 梯度存在: {grad_valid}")

        assert all(grad_exists), "存在未更新参数"
        print(f"总损失值: {loss.item():.4f}")

    # set seed
    seed = 50000
    torch.manual_seed(seed)  # CPU 上的随机数
    torch.cuda.manual_seed(seed)  # 当前 GPU 的随机数
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 的随机数（如果有多个 GPU）

    # 禁用 cuDNN 的非确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # unit tests
    test_positional_encoding_shapes()
    test_self_attention_shapes()
    test_multi_head_attention_shapes()
    test_encoder_block_shapes()

    # test
    test_encoder_decoder_data_flow()
    test_transformer_full_pipeline()
    test_transformer_end_to_end()
    print("All tests passed!")
