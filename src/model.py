import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from utils import LinearAct
from encoders import DiffModule
import math
from param import args


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim, method='bilinear'):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, pre_alpha=None, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size):
        super(SpeakerEncoder, self).__init__()
        self.feature_size = feature_size

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

    def forward(self, src, trg):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """
        if args.img_type == 'feat':
            src_feat, trg_feat = src, trg
        else:
            # Feature Extraction
            src_feat = self.resnet_extractor(src)
            trg_feat = self.resnet_extractor(trg)

            # Shape
            src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
            trg_feat = trg_feat.permute(0, 2, 3, 1)

        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1))
        trg_feat = trg_feat.view(trg_feat.size(0), -1, trg_feat.size(-1))

        # Concat
        ctx = torch.cat((src_feat, trg_feat), 1)
        # ctx = src_feat - trg_feat

        return ctx


class SpeakerDecoder(nn.Module):
    def __init__(self, ntoken, ctx_size, heads=1):
        super(SpeakerDecoder, self).__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim

        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.attention_layer = SoftDotAttention(hidden_size, ctx_size)
        self.projection = nn.Linear(hidden_size, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, self.ctx_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1) if ctx_mask is not None else None
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class SpeakerDecoderTran(nn.Module):
    def __init__(self, ntoken, ctx_size, heads=1):
        super(SpeakerDecoderTran, self).__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim

        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        self.attention_layer = TransformerAttention(
            hidden_size, ctx_size, hidden_size, merge_info=True
        )
        self.projection = nn.Linear(hidden_size, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)

        # Attention
        x = self.attention_layer(x, ctx)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class MultiCtxEncoder(nn.Module):
    def __init__(self, feature_size, normalized=False):
        super(MultiCtxEncoder, self).__init__()
        self.feature_size = feature_size

        # ResNet Feat Mean and Std
        self.normazlied = normalized
        if normalized:
            import numpy as np
            import os
            DATA_ROOT = "dataset/"
            feat_mean = np.load(os.path.join(DATA_ROOT, args.dataset, 'feat_mean.npy'))
            feat_std = np.load(os.path.join(DATA_ROOT, args.dataset, 'feat_std.npy'))
            self.feat_mean = torch.from_numpy(feat_mean).cuda()
            self.feat_std = torch.from_numpy(feat_std).cuda()

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

    @property
    def ctx_dim(self):
        return self.feature_size

    def forward(self, src, trg):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """
        if args.img_type == 'feat':
            src_feat, trg_feat = src, trg
        else:
            # Feature Extraction
            src_feat = self.resnet_extractor(src)
            trg_feat = self.resnet_extractor(trg)

            # Shape
            src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
            trg_feat = trg_feat.permute(0, 2, 3, 1)

        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1))
        trg_feat = trg_feat.view(trg_feat.size(0), -1, trg_feat.size(-1))

        # normalize
        if self.normazlied:
            src_feat = (src_feat - self.feat_mean) / self.feat_std
            trg_feat = (trg_feat - self.feat_mean) / self.feat_std

        # tuple
        ctx = (src_feat, trg_feat)

        return ctx


class TransformerAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, q_dim, k_dim, dim, merge_info=True):
        """
        :param q_dim: Dimension of Query.
        :param k_dim: Dimension of Key. (Input context)
        :param dim:   Dimension of Attention.
        :param merge_info: Whether merged infor by TanH(FC[h, h'])
        """
        super(TransformerAttention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(q_dim, dim)
        self.k_layer = nn.Linear(k_dim, dim)
        self.v_layer = nn.Linear(k_dim, dim)
        self.out_layer = nn.Linear(q_dim + dim, dim)
        self.merge_info = merge_info

    def forward(self, query, key):
        """
        :param query: [b, ..., q_dim]
        :param key:   [b, len, k_dim]
        :return:
        """
        batch_size = query.size(0)
        return_shape = query.size()[1:-1]
        q = self.q_layer(query)     # [b, ..., h_dim]
        k = self.k_layer(key)       # [b, len, h_dim]
        v = self.v_layer(key)       # [b, len, h_dim]

        q = q.view(batch_size, -1, 1, self.dim)
        k = k.view(batch_size, 1, -1, self.dim)
        l = (q * k).sum(-1) / math.sqrt(self.dim)       # b, ..., len

        p = F.softmax(l, -1)                            # b, ..., len

        # (b, ..., len) x (b, len, dim) = (b, ..., dim)
        att_vec = torch.bmm(p, v).view((batch_size,) + return_shape + (self.dim,))

        if self.merge_info:
            return torch.tanh(self.out_layer(torch.cat((att_vec, query), -1)))
        else:
            return att_vec


class NewAttEncoder(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

        if args.encoder == 'one':
            self.src2trg_att = OneAttention(feature_size, hidden_size)
            self.trg2src_att = OneAttention(feature_size, hidden_size)
        elif args.encoder == 'diff':
            self.src2trg_att = DiffModule(feature_size, hidden_size)
            self.trg2src_att = DiffModule(feature_size, hidden_size)
        else:
            assert False
        self.drop = nn.Dropout(args.dropout)
        # self.compare_fc = nn.Linear(
        #     feature_size*2, hidden_size
        # )

    @property
    def ctx_dim(self):
        if args.encoder == 'one':
            return self.feature_size
        elif args.encoder == 'diff':
            return self.hidden_size

    def forward(self, src, trg):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """
        if args.img_type == 'feat':
            src_feat, trg_feat = src, trg
        else:
            # Feature Extraction
            src_feat = self.resnet_extractor(src)
            trg_feat = self.resnet_extractor(trg)

            # Shape
            src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
            trg_feat = trg_feat.permute(0, 2, 3, 1)
        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1))
        trg_feat = trg_feat.view(trg_feat.size(0), -1, trg_feat.size(-1))

        # Attention
        src_att = self.src2trg_att(src_feat, trg_feat)
        trg_att = self.trg2src_att(trg_feat, src_feat)
        src_att = self.drop(src_att)
        trg_att = self.drop(trg_att)

        # tuple
        ctx = (src_att, trg_att)

        return ctx

class CrossAttEncoder(nn.Module):
    def __init__(self, feature_size):
        super(CrossAttEncoder, self).__init__()
        self.feature_size = feature_size
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

        self.src2trg_att = TransformerAttention(
            feature_size, feature_size, hidden_size, merge_info=True
        )
        self.trg2src_att = TransformerAttention(
            feature_size, feature_size, hidden_size, merge_info=True
        )

        # self.compare_fc = nn.Linear(
        #     feature_size*2, hidden_size
        # )

    @property
    def ctx_dim(self):
        return self.hidden_size

    def forward(self, src, trg):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """

        # Feature Extraction
        src_feat = self.resnet_extractor(src)
        trg_feat = self.resnet_extractor(trg)

        # Shape
        src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
        trg_feat = trg_feat.permute(0, 2, 3, 1)
        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1))
        trg_feat = trg_feat.view(trg_feat.size(0), -1, trg_feat.size(-1))

        # Attention
        src_att = self.src2trg_att(src_feat, trg_feat)
        trg_att = self.trg2src_att(trg_feat, src_feat)

        # Compare
        # src_compare = self.compare_fc(
        #     torch.cat(src_feat, src_feat - src_att)
        # )
        # trg_compare = self.compare_fc(
        #     torch.cat(trg_feat, trg_feat - trg_att)
        # )

        # tuple
        ctx = (src_att, trg_att)

        return ctx

class NewCtxDecoder(nn.Module):
    def __init__(self, ntoken, ctx_size, heads=1):
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
        # self.attention_layers = []
        # for _ in range(self.heads):
        #     self.attention_layers.append(
        #         TransformerAttention(hidden_size, ctx_size, hidden_size, merge_info=True).cuda()
        #     )
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
        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)                # batch_size, length, hid_dim

        # Multi-Heads Attention Layer
        # x_att = []
        # for head in range(self.heads):
        #    x_att.append(self.attention_layers[head](x, ctx[head]))    # batch_size, length, hid_dim
        # x = torch.cat(x_att, -1)

        ctx = [self.img_fc(c) for c in ctx]
        for head in range(self.heads):
            x = (self.attention_layers[head](x, ctx[head]))    # batch_size, length, hid_dim
            x = self.drop(x)

        # Output the prediction logit
        # x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1

class MultiCtxDecoder(nn.Module):
    def __init__(self, ntoken, ctx_size, heads=1):
        super(MultiCtxDecoder, self).__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim
        self.heads = heads

        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)
        # self.attention_layers = []
        # for _ in range(self.heads):
        #     self.attention_layers.append(
        #         TransformerAttention(hidden_size, ctx_size, hidden_size, merge_info=True).cuda()
        #     )
        self.attention_layers = nn.ModuleList([
            TransformerAttention(hidden_size, ctx_size, hidden_size, merge_info=True).cuda()
            for _ in range(self.heads)
        ])
        self.projection = nn.Linear(hidden_size, ntoken)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):
        """
        :param words: (batch_size, length)
        :param h0: (b x dim)
        :param c0: (b x dim)
        :param ctx:
        :param ctx_mask:
        :return:
        """
        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        # LSTM
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        x = self.drop(x)                # batch_size, length, hid_dim

        # Multi-Heads Attention Layer
        # x_att = []
        # for head in range(self.heads):
        #    x_att.append(self.attention_layers[head](x, ctx[head]))    # batch_size, length, hid_dim
        # x = torch.cat(x_att, -1)

        for head in range(self.heads):
            x = (self.attention_layers[head](x, ctx[head]))    # batch_size, length, hid_dim

        # Output the prediction logit
        x = self.drop(x)
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
        
    
class TwoAttention(nn.Module):
    def __init__(self, dim, hid_dim=None):
        super().__init__()
        if hid_dim is None:
            hid_dim = dim
        self.a2b = OneAttention(dim, hid_dim)
        self.drop = nn.Dropout(args.dropout)
        self.b2c = OneAttention(dim, hid_dim)

    def forward(self, a, b, c, debug=False):
        if debug:
            x = self.a2b(a, c)
        else:
            x = self.a2b(a, b)
            x = self.b2c(x, c)
        return x

class PostCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim, 3),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size, size, dim = x.size()
        #print(batch_size, size, dim)
        size = int(size ** 0.5)
        x = x.reshape(batch_size, size, size, dim)
        x = x.permute(0, 3, 1, 2)     # N, H, W, C --> N, C, H, W
        x = self.convs(x)
        #print(x.size())
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, -1, dim)
        return x
            
class SelfAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            LinearAct(dim, dim // 2, 'relu', bias=False),
            LinearAct(dim // 2, 1, bias=False)
        )
    
    def forward(self, x):
        mask = None
        if type(x) is tuple:
            x, mask = x
        logit = self.fc(x).squeeze(2)
        if mask is not None:
            logit.masked_fill_(mask, -float('inf'))
        prob = F.softmax(logit, -1)     # B, n
        return torch.einsum('ij,ijk->ik', prob, x)

class ScoreModel(nn.Module):
    def __init__(self, dataset, feature_size, vocab_size):
        super().__init__()
        self.dataset = dataset
        self.feature_size = feature_size
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.emb_dim = args.emb_dim

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

        # ResNet Feat Mean and Std
        self.feat_mean = torch.from_numpy(dataset.feat_mean).cuda()
        self.feat_std = torch.from_numpy(dataset.feat_std).cuda()

        # Image Encoder 
        self.pre_fc = LinearAct(self.feature_size, self.hidden_size, 'relu')

        # Lang Encoder
        self.w_emb = nn.Embedding(vocab_size, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)

        # Attentions
        self.src2inst2trg = TwoAttention(hidden_size, hidden_size)
        self.trg2inst2src = TwoAttention(hidden_size, hidden_size)
        self.inst2src2trg = TwoAttention(hidden_size, hidden_size)

        # Post Encoder
        self.post_cnn = PostCNN(hidden_size)
        self.post_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.reduce_inst = SelfAtt(hidden_size)

        # Decision
        self.classify = nn.Sequential(
            LinearAct(hidden_size * 3, hidden_size, 'tanh'),
            self.drop,
            LinearAct(hidden_size, 1, bias=False)
        )

    def forward(self, src, trg, inst, leng):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """
        batch_size = src.size(0)
        
        # Feature Extraction
        if args.img_type == 'feat':
            src_feat, trg_feat = src, trg
        else:
            src_feat = self.resnet_extractor(src)
            trg_feat = self.resnet_extractor(trg)
            src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
            trg_feat = trg_feat.permute(0, 2, 3, 1)

        # Shape
        src_feat = (src_feat - self.feat_mean) / self.feat_std * 0.176
        trg_feat = (trg_feat - self.feat_mean) / self.feat_std * 0.176
        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1))
        trg_feat = trg_feat.view(trg_feat.size(0), -1, trg_feat.size(-1))
        src_feat = self.drop(self.pre_fc(src_feat))
        trg_feat = self.drop(self.pre_fc(trg_feat))
        
        # inst Encoder
        embeds = self.w_emb(inst)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)
        h0 = c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        inst_feat, (h1, c1) = self.lstm(embeds, (h0, c0))
        #inst_feat = inst_feat / 0.176
        #print('inst', inst_feat.mean(), inst_feat.std())
        inst_feat = self.drop(inst_feat)                # batch_size, length, hid_dim

        # Create Inst Mask
        inst_mask = torch.zeros((batch_size, inst.size(1)), dtype=torch.uint8)    # Byte Tensor
        for i, l in enumerate(leng):  
            inst_mask[i, l:] = 1
        inst_mask = inst_mask.cuda()

        # Cross Attentions
        inst_mask_att = inst_mask.unsqueeze(1)      # B, l --> B, 1, l
        src_ctx = self.src2inst2trg(src_feat, (inst_feat, inst_mask_att), trg_feat, debug=True)
        trg_ctx = self.trg2inst2src(trg_feat, (inst_feat, inst_mask_att), src_feat, debug=True)
        inst_ctx = self.inst2src2trg((inst_feat, inst_mask_att), src_ctx, trg_ctx) 

        # Post Processing
        src_ctx = self.post_cnn(src_ctx)
        trg_ctx = self.post_cnn(trg_ctx)
        inst_ctx, (h1, c1) = self.post_lstm(inst_ctx, (h0, c0))

        # Reduction
        src_hid, _ = src_ctx.max(1)
        trg_hid, _ = trg_ctx.max(1)
        inst_ctx = inst_ctx.clone()
        #inst_ctx = inst_ctx.masked_fill_(inst_mask.unsqueeze(-1), float('-inf'))    # Remove uselss hiddens
        #inst_hid, _ = inst_ctx.max(1)
        inst_hid = self.reduce_inst((inst_ctx, inst_mask))

        # Decision
        x = torch.cat([src_hid, trg_hid, inst_hid], 1)
        x = self.drop(x)
        logit = self.classify(x).squeeze(1)

        return logit

class ConcatModel(nn.Module):
    def __init__(self, dataset, feature_size, vocab_size):
        super().__init__()
        self.dataset = dataset
        self.feature_size = feature_size
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.emb_dim = args.emb_dim

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-2]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

        # ResNet Feat Mean and Std
        self.feat_mean = torch.from_numpy(dataset.feat_mean).cuda()
        self.feat_std = torch.from_numpy(dataset.feat_std).cuda()

        # Image Encoder 
        self.pre_fc = LinearAct(self.feature_size, self.hidden_size, 'relu')

        # Lang Encoder
        self.w_emb = nn.Embedding(vocab_size, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)

        # Decision
        self.classify = nn.Sequential(
            LinearAct(hidden_size * 3, hidden_size, 'tanh'),
            self.drop,
            LinearAct(hidden_size, 1, bias=False)
        )

    def forward(self, src, trg, inst, leng):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """
        batch_size = src.size(0)
        
        # Feature Extraction
        if args.img_type == 'feat':
            src_feat, trg_feat = src, trg
        else:
            src_feat = self.resnet_extractor(src)
            trg_feat = self.resnet_extractor(trg)
            src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
            trg_feat = trg_feat.permute(0, 2, 3, 1)

        # Shape
        src_feat = (src_feat - self.feat_mean) / self.feat_std * 0.176
        trg_feat = (trg_feat - self.feat_mean) / self.feat_std * 0.176
        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1)).mean(1)
        trg_feat = trg_feat.view(trg_feat.size(0), -1, trg_feat.size(-1)).mean(1)
        src_feat = (self.pre_fc(src_feat))
        trg_feat = (self.pre_fc(trg_feat))
        
        # inst Encoder
        embeds = self.w_emb(inst)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)
        h0 = c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        inst_feat, (h1, c1) = self.lstm(embeds, (h0, c0))
        inst_feat = self.drop(inst_feat)                # batch_size, length, hid_dim
        
        #index = list(zip(range(batch_size), leng.numpy()))
        #inst_feat = inst_feat[index]
        inst_feat = h1.squeeze(0)
        # Decision
        x = torch.cat([src_feat, trg_feat, inst_feat], 1)
        x = self.drop(x)
        logit = self.classify(x).squeeze(1)

        return logit

