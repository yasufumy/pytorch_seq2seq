from argparse import ArgumentParser
from operator import le
import logging

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import DataParallel
from ignite.trainer import Trainer, TrainingEvents
from ignite.handlers import Validate
from torchtext import data
from torchtext.datasets.translation import WMT14

from model import Seq2Seq
from data import SmallEnJa
from utils import log_training_average_nll, get_log_validation_ppl, \
    TeacherForceUpdater, TeacherForceInference, ComputeBleu, BestModelSnapshot

BOS = '<s>'
EOS = '</s>'


def reverse_sentence(xs):
    return list(reversed(xs))


def main(args, logger):
    if args.reverse:
        preprocessing = reverse_sentence
    else:
        preprocessing = None
    SRC = data.Field(init_token=BOS, eos_token=EOS, include_lengths=True,
                     preprocessing=preprocessing)
    TRG = data.Field(init_token=BOS, eos_token=EOS)

    if args.dataset == 'enja':
        train, val, test = SmallEnJa.splits(
            exts=('.en', '.ja'), fields=(SRC, TRG))
    elif args.dataset == 'wmt14':
        train, val, test = WMT14.splits(
            exts=('.en', '.de'), fields=(SRC, TRG),
            filter_pred=lambda ex: len(ex.src) <= 50 and len(ex.trg) <= 50)

    SRC.build_vocab(train.src, max_size=args.src_vocab)
    TRG.build_vocab(train.trg, max_size=args.trg_vocab)

    stoi = TRG.vocab.stoi

    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_sizes=(args.batch, args.batch * 2), repeat=False,
        sort_within_batch=True, sort_key=SmallEnJa.sort_key,
        device=args.gpu[0] if len(args.gpu) == 1 else -1)
    test_iter = data.Iterator(
        test, batch_size=1, repeat=False, sort=False, train=False,
        device=args.gpu[0] if args.gpu else -1)

    model = Seq2Seq(
        len(SRC.vocab), args.embed, args.encoder_hidden,
        len(TRG.vocab), args.embed, args.decoder_hidden, args.encoder_layers,
        not args.encoder_unidirectional, args.decoder_layers,
        SRC.vocab.stoi[SRC.pad_token], stoi[TRG.pad_token],
        args.dropout_ratio, args.attention_type)
    model.prepare_translation(stoi[TRG.init_token], stoi[TRG.eos_token],
                              TRG.vocab.itos, args.max_length)

    translate = model.translate
    if len(args.gpu) >= 2:
        model = DataParallel(model, device_ids=args.gpu, dim=1).cuda()
    elif len(args.gpu) == 1:
        model.cuda(args.gpu[0])

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        scheduler = None
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        scheduler = MultiStepLR(
            optimizer, milestones=list(range(8, 12)), gamma=0.5)

    trainer = Trainer(
        TeacherForceUpdater(
            model, optimizer, model, args.gradient_clipping),
        TeacherForceInference(model, model))
    trainer.add_event_handler(TrainingEvents.EPOCH_COMPLETED,
                              Validate(val_iter, epoch_interval=1))
    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                              log_training_average_nll, logger=logger)
    trainer.add_event_handler(TrainingEvents.VALIDATION_COMPLETED,
                              get_log_validation_ppl(val.trg), logger=logger)

    trainer.add_event_handler(TrainingEvents.TRAINING_COMPLETED,
                              ComputeBleu(model, test.trg, translate),
                              test_iter, args.best_file, logger)
    if args.best_file is not None:
        trainer.add_event_handler(TrainingEvents.VALIDATION_COMPLETED,
                                  BestModelSnapshot(model, 'ppl', 1e10, le),
                                  args.best_file, logger)
    if scheduler is not None:
        trainer.add_event_handler(TrainingEvents.EPOCH_STARTED,
                                  lambda trainer: scheduler.step())
    trainer.run(train_iter, max_epochs=args.epoch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=[], nargs='+')
    parser.add_argument('--embed', type=int, default=512)
    parser.add_argument('--src-vocab', type=int, default=4000)
    parser.add_argument('--encoder-hidden', type=int, default=512)
    parser.add_argument('--encoder-layers', type=int, default=1)
    parser.add_argument('--encoder-unidirectional',
                        action='store_true', default=False)
    parser.add_argument('--trg-vocab', type=int, default=5000)
    parser.add_argument('--decoder-hidden', type=int, default=512)
    parser.add_argument('--decoder-layers', type=int, default=1)
    parser.add_argument('--reverse', action='store_true', default=False)
    parser.add_argument(
        '--attention-type', type=str, default='concat',
        choices=('general', 'concat', 'dot'))
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout-ratio', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--gradient-clipping', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=32)
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('--best-file', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='enja',
                        choices=('enja', 'wmt14'))
    parser.add_argument('--optim', type=str, default='adam',
                        choices=('adam', 'sgd'))
    args = parser.parse_args()

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, level=logging.INFO)
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler())
        logger = logger.info
    else:
        logger = print

    main(args, logger)
