from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.distributions import Categorical

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def remove_id(ys, tag_id):
    if type(ys) is Variable:
        ys = ys.data
    pick = partial(torch.topk, k=1) if ys.is_cuda else partial(torch.max, dim=0)
    result = []
    for y in ys:
        indices = y == tag_id
        if indices.any():
            i = pick(indices)[1][0]
            y = y[:i]
        result.append(y)
    return result


class Linear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight)
        if self.bias is not None:
            nn.init.constant(self.bias, 0)


class Embedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class LSTMCell(nn.LSTMCell):
    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight_ih)
        nn.init.xavier_uniform(self.weight_hh)
        if self.bias:
            nn.init.constant(self.bias_ih, 0)
            nn.init.constant(self.bias_hh, 0)


class BaseEncoder(nn.Module):
    def init_state(self, variable, batch_size, hidden_size):
        return Variable(
            variable.data.new(batch_size, hidden_size).zero_().float(),
            volatile=not self.training)


class LSTMEncoder(BaseEncoder):
    def __init__(self, hidden_size, embeddings, dropout_ratio=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.lstm = LSTMCell(embeddings.embedding_dim, hidden_size)
        self.dropout_ratio = dropout_ratio

    def forward(self, xs):
        embs = functional.dropout(self.embeddings(xs), p=self.dropout_ratio)
        return self._recurrent(embs, lstm=self.lstm)

    def _recurrent(self, embs, lstm, reverse=False):
        length, batch_size, _ = embs.size()
        h = m = self.init_state(embs, batch_size, self.hidden_size)
        steps = range(length - 1, -1, -1) if reverse else range(length)
        hs = []
        for i in steps:
            h, m = lstm(embs[i], (h, m))
            hs.append(functional.dropout(h, p=self.dropout_ratio))
        if reverse:
            hs.reverse()
        return torch.cat(hs, 0).view(length, batch_size, -1)


class BiLSTMEncoder(LSTMEncoder):
    def __init__(self, hidden_size, embeddings, dropout_ratio=0.5):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.lstm_forward = LSTMCell(embeddings.embedding_dim, hidden_size)
        self.lstm_backward = LSTMCell(embeddings.embedding_dim, hidden_size)
        self.dropout_ratio = dropout_ratio

    def forward(self, xs):
        embs = functional.dropout(self.embeddings(xs), p=self.dropout_ratio)
        hs_forward = super()._recurrent(embs, lstm=self.lstm_forward)
        hs_backward = super()._recurrent(embs, lstm=self.lstm_backward, reverse=True)
        return torch.cat((hs_forward, hs_backward), 2)


class BaseAttention(nn.Module):
    def __init__(self, mode='soft'):
        super().__init__()

        if mode == 'soft':
            self._forward = self._soft_forward
        elif mode == 'hard':
            self._forward = self._hard_forward
        else:
            raise ValueError('mode should be soft or hard.')

    @staticmethod
    def _soft_forward(ps, hs):
        return torch.bmm(ps.unsqueeze(1), hs).squeeze(1)

    @staticmethod
    def _hard_forward(ps, hs):
        batch_size = ps.size(0)
        positions = Categorical(ps).sample()
        indices = positions.data.new(range(batch_size))
        return hs[indices, positions]

    def forward(self, query, hs):
        scores = self.score(query, hs)
        ps = functional.softmax(scores, dim=1)
        return self._forward(ps, hs)

    def score(self, query, hs):
        raise NotImplementedError


class MLPAttention(BaseAttention):
    def __init__(self, query_size, hidden_size, mode='soft'):
        super().__init__(mode)

        self.linear = Linear(query_size + hidden_size, hidden_size, bias=False)
        self.v = Linear(hidden_size, 1, bias=False)

    def score(self, query, hs):
        _, query_size = query.size()
        _, length, hidden_size = hs.size()
        query = query.unsqueeze(1).expand(-1, length, query_size)
        linear_in = torch.cat((query, hs), 2).view(-1, query_size + hidden_size)
        return self.v(torch.tanh(self.linear(linear_in))).view(-1, length)


class GeneralAttention(BaseAttention):
    def __init__(self, query_size, hidden_size, mode='soft'):
        super().__init__(mode)

        self.linear = Linear(query_size, hidden_size, bias=False)

    def score(self, query, hs):
        return torch.bmm(hs, self.linear(query).unsqueeze(2)).squeeze(2)


class DotAttention(BaseAttention):
    def __init__(self, query_size, hidden_size, mode='hard'):
        super().__init__(mode)

        if query_size != hidden_size:
            raise ValueError('query_size and hidden_size should be equal')

    def score(self, query, hs):
        return torch.bmm(hs, query.unsqueeze(2)).squeeze(2)


class LSTMDecoder(BaseEncoder):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, embeddings,
                 attention, dropout_ratio=0.5):
        super().__init__()

        self.embeddings = embeddings
        self.attention = attention
        self.hidden_size = decoder_hidden_size
        self.lstm = LSTMCell(
            embeddings.embedding_dim + decoder_hidden_size, decoder_hidden_size)
        self.linear = Linear(
            decoder_hidden_size + encoder_hidden_size, decoder_hidden_size)
        self.linear_out = Linear(decoder_hidden_size, embeddings.num_embeddings)
        self.dropout_ratio = dropout_ratio

    def forward_step(self, y, h, m, o, hs):
        emb = functional.dropout(self.embeddings(y), p=self.dropout_ratio)
        h, m = self.lstm(torch.cat((emb, o), 1), (h, m))
        h = functional.dropout(h, p=self.dropout_ratio)
        c = self.attention(h, hs)
        o = functional.tanh(self.linear(torch.cat((c, h), 1)))
        return self.linear_out(h), h, m, o

    def forward(self, ts, hs):
        length, batch_size = ts.size()
        h = m = o = self.init_state(ts, batch_size, self.hidden_size)
        hs = hs.transpose(0, 1)  # transpose for attention
        ys = []
        for t in ts:
            y, h, m, o = self.forward_step(t, h, m, o, hs)
            ys.append(y)
        ys = torch.cat(ys, 0)
        return functional.log_softmax(ys, dim=1).view(length, batch_size, -1)


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_size, source_embed_size,  encoder_hidden_size,
                 target_vocab_size, target_embed_size, decoder_hidden_size,
                 encoder_pad_index, decoder_pad_index, dropout_ratio=0.5,
                 attention_type='general'):
        super().__init__()

        source_embedding = Embedding(
            source_vocab_size, source_embed_size, padding_idx=encoder_pad_index)
        self.encoder = BiLSTMEncoder(
            encoder_hidden_size, source_embedding, dropout_ratio)
        target_embedding = Embedding(
            target_vocab_size, target_embed_size, padding_idx=decoder_pad_index)
        if attention_type == 'general':
            attention = GeneralAttention(decoder_hidden_size, 2 * encoder_hidden_size)
        elif attention_type == 'mlp':
            attention = MLPAttention(decoder_hidden_size, 2 * encoder_hidden_size)
        elif attention_type == 'dot':
            attention = DotAttention(decoder_hidden_size, 2 * encoder_hidden_size)
        else:
            raise ValueError('attention_type should be "general" or "mlp" or "dot".')
        self.decoder = LSTMDecoder(2 * encoder_hidden_size, decoder_hidden_size,
                                   target_embedding, attention, dropout_ratio)
        self.nll_loss = nn.NLLLoss(ignore_index=decoder_pad_index)

    def compute_loss(self, batch):
        xs, ts = batch
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
        y = Variable(
            xs.data.new(batch_size).fill_(self.bos_id), volatile=True)
        h = m = o = self.decoder.init_state(
            xs, batch_size, self.decoder.hidden_size)
        ys = []
        # decoding
        for _ in range(self.max_length):
            y, h, m, o = self.decoder.forward_step(y, h, m, o, hs)
            y = y.topk(1)[1].view(-1)
            if (y.data == self.eos_id).all():
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
            hyp += [y.tolist() for y in self.translate(xs, as_string=False)]
            ts = ts.transpose(0, 1)
            ts = remove_id(ts, self.eos_id)
            ref += [[t.tolist()] for t in ts]
        return corpus_bleu(
            ref, hyp, smoothing_function=SmoothingFunction().method1) * 100
