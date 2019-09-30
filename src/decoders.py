from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
# from attentions import DynamicAttention, DynamicAttention1
from attentions import *
import math
from param import args

class DynamicDecoder(nn.Module):
    def __init__(self, ntoken, ctx_size):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim

        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.att = DynamicAttention51(
            query_dim=hidden_size,
            feat_dim=ctx_size,
            hid_dim=hidden_size,
            out_dim=hidden_size
        )
        self.projection = LinearAct(hidden_size, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        src, trg = ctx

        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)

        # Attention
        x = self.att(x, src, trg)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class DynamicDecoderFC(nn.Module):
    def __init__(self, ntoken, ctx_size):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim

        self.img_fc = LinearAct(ctx_size, hidden_size, 'tanh')
        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.att = DynamicAttention2(
            query_dim=hidden_size,
            feat_dim=hidden_size,
            hid_dim=hidden_size,
            out_dim=hidden_size
        )
        self.projection = LinearAct(hidden_size, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        ctx = [self.img_fc(c) for c in ctx]
        src, trg = ctx

        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)

        # Attention
        x = self.att(x, src, trg)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1

class DynamicDecoderMHC(nn.Module):
    def __init__(self, ntoken, ctx_size, heads=2):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim
        self.heads = heads

        self.img_fc = LinearAct(ctx_size, hidden_size, 'tanh')
        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.att = DynamicAttention2(
            query_dim=hidden_size,
            feat_dim=hidden_size,
            hid_dim=hidden_size,
            out_dim=hidden_size
        )
        self.attention_layers = nn.ModuleList([
            OneAttention(hidden_size, hidden_size)
            for _ in range(self.heads)
        ])
        self.projection = LinearAct(hidden_size * 2, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        ctx = [self.img_fc(c) for c in ctx]
        src, trg = ctx

        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)

        h = x

        # Attention
        for head in range(self.heads):
            x = (self.attention_layers[head](x, ctx[head]))    # batch_size, length, hid_dim
            x = self.drop(x)

        # Relationship
        r = self.att(h, src, trg)
        r = self.drop(r)

        # Output the prediction logit
        logit = self.projection(torch.cat((x, r), -1))

        return logit, h1, c1


class DynamicDecoderMH(nn.Module):
    def __init__(self, ntoken, ctx_size, heads=2):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim
        self.heads = heads

        self.img_fc = LinearAct(ctx_size, hidden_size, 'tanh')
        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.att = DynamicAttention5(
            query_dim=hidden_size,
            feat_dim=hidden_size,
            hid_dim=hidden_size,
            out_dim=hidden_size
        )
        self.attention_layers = nn.ModuleList([
            OneAttention(hidden_size, hidden_size)
            for _ in range(self.heads)
        ])
        self.projection = LinearAct(hidden_size, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        ctx = [self.img_fc(c) for c in ctx]
        src, trg = ctx

        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)

        # Attention
        for head in range(self.heads):
            x = (self.attention_layers[head](x, ctx[head]))    # batch_size, length, hid_dim
            x = self.drop(x)
        x = self.att(x, src, trg)
        x = self.drop(x)

        # Output the prediction logit
        logit = self.projection(x)

        return logit, h1, c1


class OneAttention(nn.Module):
    def __init__(self, dim, hid_dim=None):
        super().__init__()
        if hid_dim is None:
            hid_dim = dim
        self.dim = dim
        self.hid_dim = hid_dim
        self.q_fc = LinearAct(dim, hid_dim, bias=False)
        self.k_fc = LinearAct(dim, hid_dim, bias=False)
        self.mix = LinearAct(3*dim, dim, 'relu', bias=False)

    def forward(self, query, key):
        if type(query) is tuple:
            query = query[0]
        key_mask = None
        if type(key) is tuple:
            key, key_mask = key
        q = self.q_fc(query)    # B, l, d
        k = self.k_fc(key)      # B, n, d
        logit = torch.einsum('ijd,ikd->ijk', q, k)  # B, l, n
        logit = logit / math.sqrt(self.hid_dim)
        if key_mask is not None:
            logit.masked_fill_(key_mask, -float('inf'))
        prob = F.softmax(logit, -1) # B, l, n --> B, l, n
        ctx = torch.bmm(prob, key)  # B, l, n * B, n, d --> B, l, d
        x = torch.cat([query, ctx, query * ctx], -1)
        return self.mix(x)
