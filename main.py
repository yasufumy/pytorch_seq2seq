import re
import spacy
from argparse import ArgumentParser

import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
from torchtext import data
from torchtext import datasets

from model import Seq2Seq


parser = ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()


spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en, init_token='<s>', eos_token='</s>')

train, val, test = datasets.Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))


DE.build_vocab(train.src, min_freq=3)
EN.build_vocab(train.trg, min_freq=3)

if args.gpu >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)


train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=args.batch, device=args.gpu)

stoi = EN.vocab.stoi
model = Seq2Seq(
    len(DE.vocab), 200, 100, 100, len(EN.vocab), 200,
    DE.vocab.stoi[DE.pad_token], stoi[EN.pad_token])
model.prepare_translation(stoi[EN.init_token], stoi[EN.eos_token], EN.vocab.itos, 50)
if torch.cuda.is_available():
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=1)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
optimizer = optim.Adam(model.parameters())

for i in range(1, args.epoch + 1):
    sum_loss = 0
    print('epoch: {}'.format(i))
    # scheduler.step()
    for batch in train_iter:
        optimizer.zero_grad()
        loss = model.compute_loss(batch.src, batch.trg)
        loss.backward()
        optimizer.step()
        sum_loss += loss.data[0]
        if train_iter.epoch >= i:
            break
    print('loss: {}'.format(sum_loss))
    hyps = model.translate(batch.src)
    print('ref: {}'.format(' '.join([EN.vocab.itos[t.data[0]] for t in batch.trg][1:])))
    print('hyp: {}'.format(hyps[0]))
    print('bleu: {}'.format(model.evaluate(val_iter)))
