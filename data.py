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
