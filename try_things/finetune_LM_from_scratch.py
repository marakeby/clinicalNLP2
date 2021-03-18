from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFKC, Sequence, BertNormalizer, NFD, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, BertPreTokenizer

from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Whitespace

# tokenizer = Tokenizer(WordPiece())

# tokenizer = WordPiece()

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

bert_tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

bert_tokenizer.pre_tokenizer = Whitespace()

bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

trainer = WordPieceTrainer(
    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

import datasets

dataset = datasets.load_dataset(
    "wikitext", "wikitext-2-raw-v1", split="train+test+validation"
)


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

print (bert_tokenizer)

print (len(dataset))
bert_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))

print (bert_tokenizer)

# bert_tokenizer.train(files, trainer)
# tokenizer.normalizer = Sequence([NFKC(),Lowercase(), BertNormalizer])

# tokenizer.pre_tokenizer = BertPreTokenizer()


# tokenizer.decoder = WordPieceDecoder()

# trainer = WordPieceTrainer(vocab_size=25000, show_progress=True)

# trainer.train()