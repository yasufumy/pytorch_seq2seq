from argparse import ArgumentParser

import torch
import torch.optim as optim
from torchtext import data

from model import Seq2Seq
from data import SmallEnJa


parser = ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--embed', type=int, default=512)
parser.add_argument('--encoder-hidden', type=int, default=512)
parser.add_argument('--decoder-hidden', type=int, default=512)
parser.add_argument('--attention-type', type=str, default='general',
                    choices=('general', 'mlp', 'dot'))
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--dropout-ratio', type=float, default=0.5)
parser.add_argument('--weight-decay', type=float, default=1e-6)
parser.add_argument('--gradient-clipping', type=int, default=5)
parser.add_argument('--evaluation-step', type=int, default=3)
parser.add_argument('--max-length', type=int, default=32)
args = parser.parse_args()


EN = data.Field(eos_token='</s>')
JA = data.Field(init_token='<s>', eos_token='</s>')

train, val, test = SmallEnJa.splits(exts=('.en', '.ja'), fields=(EN, JA))


EN.build_vocab(train.src, max_size=4000)
JA.build_vocab(train.trg, max_size=5000)


train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(args.batch, 512, 512), device=args.gpu)

stoi = JA.vocab.stoi
model = Seq2Seq(
    len(EN.vocab), args.embed, args.encoder_hidden,
    len(JA.vocab), args.embed, args.decoder_hidden,
    EN.vocab.stoi[EN.pad_token], stoi[JA.pad_token], args.dropout_ratio,
    args.attention_type)
model.prepare_translation(
    stoi[JA.init_token], stoi[JA.eos_token], JA.vocab.itos, args.max_length)

if args.gpu >= 0:
    model.cuda(args.gpu)

optimizer = optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

max_bleu = -1
sum_loss = 0
max_epoch = args.epoch
eval_step = args.evaluation_step
previous_epoch = 0.

print('start training')
for batch in train_iter:
    optimizer.zero_grad()
    loss = model.compute_loss((batch.src, batch.trg))
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clipping)
    optimizer.step()
    sum_loss += loss.data[0]
    del loss, batch
    # report loss
    if train_iter.epoch // 1 != previous_epoch // 1:
        print('epoch: {}'.format(train_iter.epoch))
        print('loss: {}'.format(sum_loss))
        sum_loss = 0
    # evaluation
    if train_iter.epoch // eval_step != previous_epoch // eval_step:
        # evaluation model
        model.eval()
        bleu = model.evaluate(val_iter)
        print('bleu: {}'.format(bleu))
        if bleu > max_bleu:
            max_bleu = bleu
            torch.save(model.state_dict(), 'best.pt')
        # back to training mode
        model.train()
    # stop training
    if train_iter.epoch // max_epoch != previous_epoch // max_epoch:
            break
    previous_epoch = train_iter.epoch

model.load_state_dict(torch.load('best.pt'))
model.eval()
print('test score: {}'.format(model.evaluate(test_iter)))
