import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def make_zeros(*shape, **kwargs):
    volatile = kwargs.pop('volatile', False)
    zeros = Variable(torch.zeros(*shape), volatile=volatile)
    if torch.cuda.is_available():
        zeros = zeros.cuda()
    return zeros


def remove_id(ys, tag_id):
    if type(ys) is Variable:
        ys = ys.data
    result = []
    for y in ys:
        indices = y == tag_id
        if any(indices):
            # i = indices.max(0)[1][0]
            i = indices.topk(1)[1][0]  # use topk insted of max
            y = y[:i]
        result.append(y)
    return result


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embeddings):
        super().__init__()

        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.lstm = nn.LSTMCell(embeddings.embedding_dim, hidden_size)

    def forward(self, xs, rnn_name=None):
        if rnn_name is None:
            rnn = getattr(self, 'lstm')
        else:
            rnn = getattr(self, rnn_name)
        length, batch_size = xs.size()
        h = m = make_zeros(batch_size, self.hidden_size)
        hs = []
        for x in xs:
            emb = self.embeddings(x)
            h, m = rnn(emb, (h, m))
            hs.append(h)
        return torch.cat(hs, 0).view(length, batch_size, -1)


class BiLSTMEncoder(LSTMEncoder):
    def __init__(self, hidden_size, embeddings):
        super().__init__(hidden_size, embeddings)

        # for backward
        self.lstm2 = nn.LSTMCell(embeddings.embedding_dim, hidden_size)

    def forward(self, xs):
        hs_forward = super().forward(xs)
        inverse_index = torch.arange(xs.size(0) - 1, -1, -1).long()
        if torch.cuda.is_available():
            inverse_index = inverse_index.cuda()
        hs_backward = super().forward(xs[inverse_index], 'lstm2')[inverse_index]
        return torch.cat((hs_forward, hs_backward), 2)


class MLPAttention(nn.Module):
    def __init__(self, query_size, hidden_size):
        super().__init__()

        self.linear = nn.Linear(query_size + hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, hs):
        ps = self.align(query, hs)
        return torch.bmm(ps, hs).view(query.size(0), -1)

    def align(self, query, hs):
        batch_size, query_size = query.size()
        batch_size, length, hidden_size = hs.size()
        query = query.view(batch_size, 1, query_size).expand(
            batch_size, length, query_size)
        linear_in = torch.cat((query, hs), 2).view(-1, query_size + hidden_size)
        score = self.v(torch.tanh(self.linear(linear_in))).view(batch_size, length)
        return functional.softmax(score).view(batch_size, 1, length)


class LSTMDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, embeddings, attention):
        super().__init__()

        self.embeddings = embeddings
        self.attention = attention
        self.hidden_size = decoder_hidden_size
        self.lstm = nn.LSTMCell(
            embeddings.embedding_dim + decoder_hidden_size, decoder_hidden_size)
        self.linear = nn.Linear(
            decoder_hidden_size + encoder_hidden_size, decoder_hidden_size)
        self.linear_out = nn.Linear(decoder_hidden_size, embeddings.num_embeddings)

    def forward_step(self, y, h, m, o, hs):
        emb = self.embeddings(y)
        h, m = self.lstm(torch.cat((emb, o), 1), (h, m))
        c = self.attention(h, hs)
        o = functional.tanh(self.linear(torch.cat((c, h), 1)))
        return self.linear_out(h), h, m, o

    def forward(self, ts, hs):
        length, batch_size = ts.size()
        h = m = o = make_zeros(batch_size, self.hidden_size)
        hs = hs.transpose(0, 1)  # transpose for attention
        ys = []
        for t in ts:
            y, h, m, o = self.forward_step(t, h, m, o, hs)
            ys.append(y)
        ys = torch.cat(ys, 0)
        return functional.log_softmax(ys).view(length, batch_size, -1)


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_size, source_embed_size,  encoder_hidden_size,
                 decoder_hidden_size, target_vocab_size, target_embed_size,
                 encoder_pad_index, decoder_pad_index):
        super().__init__()

        source_embedding = nn.Embedding(
            source_vocab_size, source_embed_size, padding_idx=encoder_pad_index)
        self.encoder = BiLSTMEncoder(encoder_hidden_size, source_embedding)
        target_embedding = nn.Embedding(
            target_vocab_size, target_embed_size, padding_idx=decoder_pad_index)
        attention = MLPAttention(decoder_hidden_size, 2 * encoder_hidden_size)
        self.decoder = LSTMDecoder(2 * encoder_hidden_size, decoder_hidden_size,
                                   target_embedding, attention)
        self.nll_loss = nn.NLLLoss(ignore_index=decoder_pad_index)

    def compute_loss(self, xs, ts):
        hs = self.encoder(xs)
        ts_in = ts[:-1]  # ignore eos
        ys = self.decoder(ts_in, hs)
        ys = ys.view(-1, ys.size(2))
        ts_out = ts[1:].view(-1)  # ignore bos
        return self.nll_loss(ys, ts_out)

    def prepare_translation(self, bos_id, eos_id, id_to_token, max_length):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.id_to_token = id_to_token
        self.max_length = max_length

    def translate(self, xs, as_string=True):
        batch_size = xs.size(1)
        hs = self.encoder(xs)
        hs = hs.transpose(0, 1)  # transpose for attention
        y = Variable(torch.LongTensor([self.bos_id] * batch_size), volatile=True)
        if torch.cuda.is_available():
            y = y.cuda()
        h = m = o = make_zeros(batch_size, self.decoder.hidden_size, volatile=True)
        ys = []
        for _ in range(self.max_length):
            y, h, m, o = self.decoder.forward_step(y, h, m, o, hs)
            y = y.topk(1)[1].view(-1)
            if all(y.data == self.eos_id):
                break
            ys.append(y.data)
        # transpose
        ys = torch.cat(ys, 0).view(-1, batch_size).transpose(0, 1)
        # remove eos id
        ys = remove_id(ys, self.eos_id)
        if as_string:
            # convert to readable sentences
            id_to_token = self.id_to_token
            sentences = [' '.join([id_to_token[i] for i in y]) for y in ys]
            return sentences
        else:
            return ys

    def evaluate(self, val_iter):
        hyp = []
        ref = []
        for batch in val_iter:
            xs = batch.src
            ts = batch.trg
            hyp += self.translate(xs, as_string=False)
            ts = ts.transpose(0, 1)
            ts = remove_id(ts, self.eos_id)
            ref += [[t] for t in ts]
        return corpus_bleu(
            ref, hyp, smoothing_function=SmoothingFunction().method1) * 100
