from torchtext import data
from torchtext.datasets import TranslationDataset


class SmallEnJa(TranslationDataset):
    urls = ['https://github.com/odashi/small_parallel_enja/archive/master.zip']
    name = 'smallenja'
    dirname = 'small_parallel_enja-master'

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='dev',
               test='test', **kwargs):
        return super(SmallEnJa, cls).splits(
            exts, fields, root, train, validation, test, **kwargs)


if __name__ == '__main__':
    EN = data.Field()
    JA = data.Field()

    train, val, test = SmallEnJa.splits(exts=('.en', '.ja'), fields=(EN, JA))

    EN.build_vocab(train.src, min_freq=3)
    JA.build_vocab(train.trg, min_freq=3)

    train_iter, val_iter = data.BucketIterator.splits(
        (train, val), batch_size=3, device=0)
