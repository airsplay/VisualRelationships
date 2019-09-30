from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
from param import args

class DynamicAttention(nn.Module):
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = self.src_linear

        self.out_linear = LinearAct(feat_dim * 2 + query_dim, out_dim)

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, _ = query.size()
        _, s, _ = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        weight = torch.einsum("ijk,isk,itk->ijst", q, s_key, t_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_ctx = torch.einsum('ijst,isf->ijf', weight, src)
        t_ctx = torch.einsum('ijst,itf->ijf', weight, trg)

        x = torch.cat((query, s_ctx, t_ctx), -1)
        x = self.out_linear(x)

        return x

class DynamicAttention1(nn.Module):
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = self.src_linear

        self.out_linear = LinearAct(hid_dim + query_dim, out_dim)

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, _ = query.size()
        _, s, _ = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        diff_key = (s_key.unsqueeze(2) - t_key.unsqueeze(1)) / math.sqrt(2)
        weight = torch.einsum("ijk,istk->ijst", q, diff_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        diff_ctx = torch.einsum('ijst,istk->ijk', weight, diff_key)

        x = torch.cat((query, diff_ctx), -1)
        x = self.out_linear(x)

        return x


class DynamicAttention2(nn.Module):
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim, bias=False)
        self.src_linear = LinearAct(feat_dim, hid_dim, bias=False)
        self.trg_linear = self.src_linear

        self.out_linear = LinearAct(hid_dim * 4 + query_dim, out_dim, act='relu',
                                    bias=False)

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, _ = query.size()
        _, s, _ = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        weight = torch.einsum("ijk,isk,itk->ijst", q, s_key, t_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_ctx = torch.einsum('ijst,isf->ijf', weight, s_key)
        t_ctx = torch.einsum('ijst,itf->ijf', weight, t_key)

        x = torch.cat((query, s_ctx, t_ctx, query*s_ctx, query*t_ctx), -1)
        x = self.out_linear(x)

        return x

class DynamicAttention21(nn.Module):
    """
    A purely productive attention
    """
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = LinearAct(feat_dim, hid_dim)

        self.src_value_linear = LinearAct(feat_dim, hid_dim)
        self.trg_value_linear = LinearAct(feat_dim, hid_dim)

        self.out_linear = LinearAct(hid_dim * 1 + query_dim, out_dim, act='relu')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, h = query.size()
        _, s, f = src.size()
        _, t, _ = trg.size()

        # Pad Zero
        # pad_tensor = torch.zeros(b, 1, f).cuda()
        # src = torch.cat((src, pad_tensor), 1)
        # trg = torch.cat((trg, pad_tensor), 1)
        q = self.query_linear(query)        # b, l, d
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        weight = torch.einsum("ijk,isk,itk->ijst", q, s_key, t_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_value = self.src_value_linear(src)            # b, s, d
        t_value = self.trg_value_linear(trg)            # b, t, d
        ctx = torch.einsum('ijst, isk, itk->ijk', weight, s_value, t_value)

        x = torch.cat((query, ctx), -1)
        x = self.out_linear(x)

        return x


class DynamicAttention3(nn.Module):
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = LinearAct(feat_dim, hid_dim)

        self.out_linear = LinearAct(hid_dim * 2 + query_dim, out_dim, act='tanh')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, h = query.size()
        _, s, f = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d
        pad_tensor = torch.zeros(b, 1, f).cuda()
        src = torch.cat((src, pad_tensor), 1)
        trg = torch.cat((trg, pad_tensor), 1)
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        weight = torch.einsum("ijk,isk,itk->ijst", q, s_key, t_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s+1, t+1)

        s_ctx = torch.einsum('ijst,isf->ijf', weight, s_key)
        t_ctx = torch.einsum('ijst,itf->ijf', weight, t_key)

        x = torch.cat((query, s_ctx, t_ctx), -1)
        x = self.out_linear(x)

        return x


class DynamicAttention4(nn.Module):
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = LinearAct(feat_dim, hid_dim)

        self.out_linear = LinearAct(hid_dim + query_dim, out_dim, act='relu')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, h = query.size()
        _, s, f = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d
        # pad_tensor = torch.zeros(b, 1, f).cuda()
        # src = torch.cat((src, pad_tensor), 1)
        # trg = torch.cat((trg, pad_tensor), 1)
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        rel_key = torch.tanh(s_key.unsqueeze(1) + t_key.unsqueeze(2))
        weight = torch.einsum("ijk,istk->ijst", q, rel_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        rel_ctx = torch.einsum('ijst,istk->ijk', weight, rel_key)

        x = torch.cat((query, rel_ctx), -1)
        x = self.out_linear(x)

        return x

class DynamicAttention5(nn.Module):
    """
    A purely additive attention
    """
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = LinearAct(feat_dim, hid_dim)

        self.src_value_linear = LinearAct(feat_dim, hid_dim)
        self.trg_value_linear = LinearAct(feat_dim, hid_dim)

        self.out_linear = LinearAct(hid_dim * 2 + query_dim, out_dim, act='relu')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, h = query.size()
        _, s, f = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d

        # Pad Zero
        # pad_tensor = torch.zeros(b, 1, f).cuda()
        # src = torch.cat((src, pad_tensor), 1)
        # trg = torch.cat((trg, pad_tensor), 1)

        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        rel_key = (s_key.unsqueeze(1) + t_key.unsqueeze(2)) / 2
        weight = torch.einsum("ijk,istk->ijst", q, rel_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_value = self.src_value_linear(src)            # b, s, d
        t_value = self.trg_value_linear(trg)            # b, t, d
        s_ctx = torch.einsum('ijst,isf->ijf', weight, s_value)
        t_ctx = torch.einsum('ijst,itf->ijf', weight, t_value)

        x = torch.cat((query, s_ctx, t_ctx), -1)
        x = self.out_linear(x)

        return x

class DynamicAttention51(nn.Module):
    """
    A purely additive attention
    """
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = LinearAct(feat_dim, hid_dim)

        self.src_value_linear = LinearAct(feat_dim, hid_dim)
        self.trg_value_linear = LinearAct(feat_dim, hid_dim)

        self.out_linear = LinearAct(hid_dim + query_dim, out_dim, act='relu')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, h = query.size()
        _, s, f = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d

        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        rel_key = torch.tanh((s_key.unsqueeze(1) + t_key.unsqueeze(2)) / 2)
        weight = torch.einsum("ijk,istk->ijst", q, rel_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_value = self.src_value_linear(src)            # b, s, d
        t_value = self.trg_value_linear(trg)            # b, t, d
        rel_value = torch.tanh((s_value.unsqueeze(1) + t_value.unsqueeze(2)) / 2)
        ctx = torch.einsum('ijst,istk->ijk', weight, rel_value)

        x = torch.cat((query, ctx), -1)
        x = self.out_linear(x)

        return x

class DynamicAttention52(nn.Module):
    """
    A purely additive attention
    """
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = LinearAct(feat_dim, hid_dim)

        self.src_value_linear = LinearAct(feat_dim, hid_dim)
        self.trg_value_linear = LinearAct(feat_dim, hid_dim)

        self.out_linear = LinearAct(hid_dim + query_dim, out_dim, act='relu')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, h = query.size()
        _, s, f = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d

        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d
        rel_key = torch.tanh((s_key.unsqueeze(1) + t_key.unsqueeze(2)) / 2)
        weight = torch.einsum("ijk,istk->ijst", q, rel_key) / math.sqrt(self.hid_dim)

        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_value = self.src_value_linear(src)            # b, s, d
        t_value = self.trg_value_linear(trg)            # b, t, d
        rel_value = torch.tanh((s_value.unsqueeze(1) + t_value.unsqueeze(2)) / 2)
        ctx = torch.einsum('ijst,istk->ijk', weight, rel_value)

        x = torch.cat((query, ctx), -1)
        x = self.out_linear(x)

        return x


class DynamicAttention7(nn.Module):
    def __init__(self, query_dim, feat_dim, hid_dim, out_dim=None):
        super().__init__()

        self.hid_dim = hid_dim
        if out_dim is None:
            out_dim = hid_dim

        self.query_linear = LinearAct(query_dim, hid_dim)
        self.src_linear = LinearAct(feat_dim, hid_dim)
        self.trg_linear = self.src_linear

        # self.alpha = 1.
        self.alpha = nn.Parameter(torch.ones(1))

        self.out_linear = LinearAct(hid_dim * 2 + query_dim, out_dim, act='relu')

    def forward(self, query, src, trg):
        """
        :param hidden: b, l, h
        :param src:    b, s, f
        :param trg:    b, t, f
        :return:
        """
        b, l, _ = query.size()
        _, s, _ = src.size()
        _, t, _ = trg.size()

        q = self.query_linear(query)        # b, l, d
        s_key = self.src_linear(src)            # b, s, d
        t_key = self.trg_linear(trg)            # b, t, d

        pad_tensor = torch.ones(b, 1, self.hid_dim).cuda() * self.alpha
        s_key = torch.cat((s_key, pad_tensor), 1)
        t_key = torch.cat((t_key, pad_tensor), 1)

        weight = torch.einsum("ijk,isk,itk->ijst", q, s_key, t_key) / math.sqrt(self.hid_dim)

        b, l, s, t = weight.size()
        weight = F.softmax(weight.view(b, l, -1), dim=-1).view(b, l, s, t)

        s_ctx = torch.einsum('ijst,isf->ijf', weight, s_key)
        t_ctx = torch.einsum('ijst,itf->ijf', weight, t_key)

        x = torch.cat((query, s_ctx, t_ctx), -1)
        x = self.out_linear(x)

        return x

