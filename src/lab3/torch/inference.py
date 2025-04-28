import torch
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from data_process import src_vocab, trg_vocab, test_loader  # 导入数据模块
from model import Transformer, Encoder, Decoder  # 导入模型定义
from tqdm import tqdm


class Translator:
    def __init__(self, model_path, src_vocab, trg_vocab):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # 检查 Apple Silicon GPU 是否可用
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']

        self.model = Transformer(
            encoder=Encoder(
                src_vocab_size=len(src_vocab.word2idx),
                embedding_size=self.config['d_model'],
                head_size=self.config['n_head'],
                ffn_size=self.config['ffn_size'],
                num_blocks=self.config['num_blocks'],
                p=0.0  # 推理时关闭dropout
            ),
            decoder=Decoder(
                trg_vocab_size=len(trg_vocab.word2idx),
                embedding_size=self.config['d_model'],
                head_size=self.config['n_head'],
                ffn_size=self.config['ffn_size'],
                num_blocks=self.config['num_blocks'],
                p=0.0
            )
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def _prepare_input(self, src_seq):
        """处理输入序列"""
        src_tensor = torch.tensor([2] + src_seq + [3]).to(self.device)

        # 添加batch维度并填充
        src_tensor = src_tensor.unsqueeze(0)  # [1, seq_len]
        return src_tensor

    def translate(self, src_seq, max_length=50):
        """使用贪心算法进行翻译"""
        src_tensor = self._prepare_input(src_seq)

        # 初始化decoder输入
        decoder_input = torch.tensor([[2]]).to(self.device)

        # 自回归生成
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(
                    encoder_input=src_tensor,
                    decoder_input=decoder_input,
                    src_who_is_pad=0,
                    trg_who_is_pad=0
                )

            # 获取最后一个token的预测
            # print(logits)
            next_token = logits[-1, :].argmax(-1)
            # print(next_token)
            # print()
            # print(f'decoder input: {decoder_input}')
            decoder_input = torch.cat(
                [decoder_input, next_token.unsqueeze(0).unsqueeze(0)], dim=-1
            )

            # 遇到EOS则停止，同时拼接EOS
            if next_token.item() == 3:
                decoder_input = torch.cat(
                    [decoder_input, torch.tensor(3).unsqueeze(0).unsqueeze(0).to(self.device)], dim=-1
                )
                break

        # 转换为token列表
        output_tokens = decoder_input[0].cpu().tolist()

        # 去除特殊token并解码
        filtered = [t for t in output_tokens if t not in {2, 3, 0}]

        return self.trg_vocab.decode(filtered)

    def calculate_bleu(self, test_loader_):
        """计算整个测试集的BLEU分数"""
        references, hypotheses = [], []

        for batch in tqdm(test_loader_, desc="Calculating BLEU"):
            src_batch, trg_batch = batch

            # 处理每个样本
            for src_seq, trg_seq in zip(src_batch, trg_batch):
                ref = [self.trg_vocab.decode([t for t in trg_seq if t not in {0}])]

                hyp = self.translate(src_seq)

                references.append(ref)
                hypotheses.append(hyp)

        return corpus_bleu(references, hypotheses)


if __name__ == "__main__":
    # 初始化翻译器
    translator = Translator(
        model_path='ckpts/best_model.pt',
        src_vocab=src_vocab,
        trg_vocab=trg_vocab
    )

    # 计算 BLEU 分数
    if True:
        bleu_score = translator.calculate_bleu(test_loader)
        print(f"\nBLEU Score: {bleu_score:.4f}")

    # 示例翻译 - 德语到英语，包含标准答案
    example_src_with_refs = [
        ("Die Katze ist auf dem Tisch .", "The cat is on the table."),
        ("Sie liest jeden Tag ein Buch .", "She reads a book every day."),
        ("Der Hund spielt im Garten .", "The dog is playing in the garden."),
        ("Wir gehen ins Kino heute Abend .", "We are going to the cinema this evening."),
        ("Das Wetter ist sehr schön heute .", "The weather is very nice today."),
        ("Er hat eine rote Jacke an .", "He is wearing a red jacket."),
        ("Ich habe Hunger .", "I am hungry."),
        ("Es gibt viele Bücher im Regal .", "There are many books on the shelf."),
        ("bitte.", "Can you help me, please?"),
        ("Morgen werde ich einkaufen gehen .", "Tomorrow I will go shopping."),
        ("Eine alte Frau sitzt an einem Webstuhl und stellt Stoff her.", "An old woman sits at a loom and makes fabric."),
        ("Ein Junge im Vordergrund blickt über andere Personen auf einem Platz.", "A boy in the foreground looks over other people on a square."),
        ("Ein Mann in einer blauen Jacke hält einen Jungen an der Schulter.", "A man in a blue jacket holds a boy by the shoulder."),
        ("Drei braune Hunde springen an der blau gekleideten Frau hoch.", "Three brown dogs jump up at the woman dressed in blue."),
        ("Ein Junge hängt aus dem Fenster eines vorbeifahrenden Taxis.", "A boy leans out of the window of a passing taxi."),
        ("Ein Mann in einem grauen Shirt springt über die Spitze einer Sanddüne in der Wüste.", "A man in a gray shirt jumps over the top of a sand dune in the desert."),
        ("Ein Kind in einem gelben Shirt springt rauf und runter.", "A child in a yellow shirt jumps up and down."),
        ("Ein kleiner Junge in einem weiß gestreiften Shirt und mit einem Stirnband hält einen Tennisschläger.", "A little boy in a white striped shirt and headband holds a tennis racket."),
        ("Zwei Hunde laufen im Wald weg von der Kamera.", "Two dogs run in the woods away from the camera."),
        ("Ein Mädchen spielt komplett bekleidet im Springbrunnen.", "A girl plays fully dressed in the fountain."),
        ("Drei junge Sumoringer stehen und hören einem Ansager zu.", "Three young sumo wrestlers stand and listen to an announcer."),
        ("Zwei männliche Wanderer inspizieren einen Baumstamm, der am Rand eines Waldweges liegt.", "Two male hikers inspect a tree trunk lying at the edge of a forest path."),
        ("Drei Spieler bringen einen gegnerischen Spieler zu Boden.", "Three players tackle an opposing player to the ground."),
        ("Zwei Jungen überqueren eine Straße in einer Stadt während sie mit einem roten Fußball spielen.", "Two boys cross a street in a city while playing with a red soccer ball."),
        ("Ein Mann hält einen anderen mit seinem Rücken hoch.", "A man holds another person up with his back."),
        ("Weißer Hund auf einem Berg dreht sich zu etwas außerhalb des Bildes um, Himmel im Hintergrund.", "A white dog on a mountain turns to something outside the picture, sky in the background."),
        ("Ein Junge fährt Skateboard auf einer Skateboardrampe.", "A boy is skateboarding on a skateboard ramp."),
        ("Ein Musiker mit langen Haaren spielt Keyboard.", "A musician with long hair plays keyboard.")
    ]

    print("\nExample Translations (German -> English):")
    for src, ref_translation in example_src_with_refs:
        # 编码源句子
        src_encoded = src_vocab.encode(src, add_special_tokens=True)

        # 生成翻译
        translation = translator.translate(src_encoded)
        translation = " ".join(translation)

        # 打印双语对照结果，包括标准答案
        print()
        print(f"Source (German)                : {src}")
        print(f"Reference Translation (English): {ref_translation}")
        print(f"Model Translation (English)    : {translation}")
