from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class LSTM(nn.LSTM):
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform(param)
            elif 'bias' in name:
                nn.init.constant(param, 0)


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embeddings, dropout_ratio=0.5,
                 bidirectional=False, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.lstm = LSTM(embeddings.embedding_dim, hidden_size,
                         dropout=dropout_ratio, bidirectional=bidirectional,
                         num_layers=num_layers)
        self.dropout_ratio = dropout_ratio

    def forward(self, xs, lengths):
        self.lstm.flatten_parameters()
        embs = functional.dropout(self.embeddings(xs), p=self.dropout_ratio)
        lengths = lengths.view(-1).tolist()
        packed_embs = pack_padded_sequence(embs, lengths)
        hs, state = self.lstm(packed_embs)
        hs = pad_packed_sequence(hs)[0]
        return hs, state


class BaseAttention(nn.Module):

    def forward(self, query, hs):
        scores = self.score(query, hs)
        scores.data.masked_fill_(self.mask, -1024)
        ps = functional.softmax(scores, dim=1)
        return torch.bmm(ps.unsqueeze(1), hs).squeeze(1)

    def score(self, query, hs):
        raise NotImplementedError

    def set_mask(self, src_lengths):
        src_lengths = src_lengths.view(-1)
        batch_size = src_lengths.numel()
        max_length = src_lengths.max()
        self.mask = torch.arange(0, max_length).\
            type_as(src_lengths).repeat(batch_size, 1).\
            ge(src_lengths.unsqueeze(1))


class ConcatAttention(BaseAttention):
    def __init__(self, query_size, hidden_size):
        super().__init__()

        self.linear = Linear(query_size + hidden_size, hidden_size, bias=False)
        self.v = Linear(hidden_size, 1, bias=False)

    def score(self, query, hs):
        _, query_size = query.size()
        _, length, hidden_size = hs.size()
        query = query.unsqueeze(1).expand(-1, length, query_size)
        linear_in = torch.cat((query, hs), 2).view(-1, query_size + hidden_size)
        return self.v(torch.tanh(self.linear(linear_in))).view(-1, length)


class GeneralAttention(BaseAttention):
    def __init__(self, query_size, hidden_size):
        super().__init__()

        self.linear = Linear(query_size, hidden_size, bias=False)

    def score(self, query, hs):
        return torch.bmm(hs, self.linear(query).unsqueeze(2)).squeeze(2)


class DotAttention(BaseAttention):
    def __init__(self, query_size, hidden_size):
        super().__init__()

        if query_size != hidden_size:
            raise ValueError('query_size and hidden_size should be equal')

    def score(self, query, hs):
        return torch.bmm(hs, query.unsqueeze(2)).squeeze(2)


class LSTMDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, embeddings, attention,
                 dropout_ratio=0.5, num_layers=1):
        super().__init__()

        self.embeddings = embeddings
        self.attention = attention
        self.hidden_size = decoder_hidden_size
        self.lstm = LSTM(embeddings.embedding_dim + decoder_hidden_size, decoder_hidden_size,
                         dropout=dropout_ratio, bidirectional=False, num_layers=num_layers)
        total_hidden_size = decoder_hidden_size + encoder_hidden_size
        self.linear = Linear(total_hidden_size, decoder_hidden_size)
        self.output = Linear(decoder_hidden_size, embeddings.num_embeddings)
        self.dropout_ratio = dropout_ratio
        self.num_layers = num_layers
        self.num_directions = 1

    def forward_step(self, y, state, hs, feed):
        emb = self.embeddings(y)
        h, state = self.lstm(
            torch.cat((emb, feed), 1).unsqueeze(0), state)
        h = functional.dropout(h.squeeze(0), p=self.dropout_ratio)
        c = self.attention(h, hs)
        feed = functional.tanh(self.linear(torch.cat((c, h), 1)))
        return self.output(feed), state, feed

    def forward(self, ts, hs, encoder_state):
        self.lstm.flatten_parameters()
        length, batch_size = ts.size()
        feed = Variable(ts.data.new(batch_size, self.hidden_size).zero_().float(),
                        volatile=not self.training)
        state = None
        hs = hs.transpose(1, 0)  # transpose for attention
        ys = []
        for t in ts:
            y, state, feed = self.forward_step(t, state, hs, feed)
            ys.append(y)
        ys = torch.cat(ys, 0)
        return functional.log_softmax(ys, dim=1).view(length * batch_size, -1)


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_size, source_embed_size, encoder_hidden_size,
                 target_vocab_size, target_embed_size, decoder_hidden_size,
                 encoder_layers, encoder_bidirectional, decoder_layers,
                 encoder_pad_index, decoder_pad_index, dropout_ratio=0.5,
                 attention_type='general'):
        super().__init__()

        source_embedding = Embedding(source_vocab_size, source_embed_size, padding_idx=encoder_pad_index)
        self.encoder = LSTMEncoder(encoder_hidden_size, source_embedding, dropout_ratio,
                                   bidirectional=encoder_bidirectional, num_layers=encoder_layers)
        target_embedding = Embedding(target_vocab_size, target_embed_size, padding_idx=decoder_pad_index)
        in_size = encoder_hidden_size * (2 if encoder_bidirectional else 1)
        if attention_type == 'general':
            attention = GeneralAttention(decoder_hidden_size, in_size)
        elif attention_type == 'concat':
            attention = ConcatAttention(decoder_hidden_size, in_size)
        elif attention_type == 'dot':
            attention = DotAttention(decoder_hidden_size, in_size)
        else:
            raise ValueError('attention_type should be "general" or "concat" or "dot".')
        self.decoder = LSTMDecoder(in_size, decoder_hidden_size, target_embedding, attention,
                                   dropout_ratio, num_layers=decoder_layers)
        self.nll_loss = nn.NLLLoss(ignore_index=decoder_pad_index, reduce=False)

    def forward(self, xs, lengths, ts):
        # encode
        hs, encoder_state = self.encoder(xs, lengths.data)
        ts_in = ts[:-1]  # ignore eos
        # decode
        self.decoder.attention.set_mask(lengths.data)
        ys = self.decoder(ts_in, hs, encoder_state)
        # loss
        ts_out = ts[1:].view(-1)  # ignore bos
        return self.nll_loss(ys, ts_out).unsqueeze(1)

    def prepare_translation(self, bos_id, eos_id, id_to_token, max_length):
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.id_to_token = id_to_token
        self.max_length = max_length

    def translate(self, xs, lengths):
        batch_size = xs.size(1)
        hs, _ = self.encoder(xs, lengths)
        hs = hs.transpose(1, 0)  # transpose for attention
        y = Variable(xs.data.new(batch_size).fill_(self.bos_id), volatile=True)
        self.decoder.attention.set_mask(lengths)
        feed = Variable(xs.data.new(batch_size, self.decoder.hidden_size).zero_().float(),
                        volatile=not self.training)
        state = None
        ys = []
        # decoding
        self.decoder.lstm.flatten_parameters()
        for _ in range(self.max_length):
            y, state, feed = self.decoder.forward_step(y, state, hs, feed)
            y = y.topk(1)[1].view(-1)
            if (y.data == self.eos_id).all():
                break
            ys.append(y.data)
        # transpose
        ys = torch.cat(ys, 0).view(-1, batch_size).transpose(0, 1)
        # remove eos id
        pick = partial(torch.topk, k=1) if ys.is_cuda else partial(torch.max, dim=0)
        eos_id = self.eos_id
        results = []
        for y in ys:
            indices = y == eos_id
            if indices.any():
                i = pick(indices)[1][0]
                y = y[:i]
            results.append(y)
        # convert to readable sentences
        id_to_token = self.id_to_token
        sentences = [[id_to_token[i] for i in y] for y in results]
        return sentences
