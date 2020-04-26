import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBase(nn.Module):
    def __init__(self, attn_type, query_dim, key_dim):
        super(AttentionBase, self).__init__()
        self.attn_type = attn_type
        self.query_dim = query_dim
        self.key_dim = key_dim

    def _score(self, query, keys):
        raise NotImplementedError

    def forward(self, query, keys, values, mask=None, return_logits=False):
        # query: target_len x b x query_dim
        # keys: b x src_len x key_dim
        # values: b x src_len x value_dim
        # mask: b x src_len, mask = 0/-inf

        weight = self._score(query, keys) # b x tgt_len x src_len
        if mask is not None:
            weight = weight + mask.unsqueeze(1).expand_as(weight)
        score = F.softmax(weight, dim=-1) # b x tgt_len x src_len
        ctx = torch.bmm(score, values) # b x tgt_len x value_dim
        if return_logits:
            return score, ctx, weight
        else:
            return score, ctx

class DotAttention(AttentionBase):
    def __init__(self, dim):
        super(DotAttention, self).__init__("dot", dim, dim)
        # not other parameters for DotAttention

    def _score(self, query, keys):
        # query: tgt_len x b x query_dim
        # keys: b x src_len x key_dim
        qdim, kdim = query.size(-1), keys.size(-1)
        # check size and dimension
        assert qdim == kdim
        assert query.dim() == 3
        assert keys.dim() == 3

        return torch.bmm(query.transpose(0, 1), keys.transpose(1, 2))

class GeneralAttention(AttentionBase):
    def __init__(self, query_dim, key_dim):
        super(GeneralAttention, self).__init__("general", query_dim, key_dim)
        self.W = nn.Linear(query_dim, key_dim, bias=False)

    def _score(self, query, keys):
        qdim, kdim = query.size(-1), keys.size(-1)
        # check size and dimension
        assert qdim == self.query_dim
        assert kdim == self.key_dim
        assert query.dim() == 3
        assert keys.dim() == 3

        return torch.bmm(self.W(query).transpose(0, 1), keys.transpose(1, 2))


class ConcatAttention(AttentionBase):
    def __init__(self, query_dim, key_dim):
        super(ConcatAttention, self).__init__("concat", query_dim, key_dim)
        self.Wa = nn.Linear(query_dim + key_dim, key_dim, bias=False)
        self.va = nn.Linear(key_dim, 1, bias=False)

    def _score(self, query, keys):
        qdim, kdim = query.size(-1), keys.size(-1)
        # check size and dimension
        assert qdim == self.query_dim
        assert kdim == self.key_dim
        assert query.dim() == 3
        assert keys.dim() == 3

        # query: target_len x batch_size x query_dim
        # keys: batch_size x src_len x key_dim

        tgt_len, src_len = query.size(0), keys.size(1)

        query = query.transpose(0, 1).unsqueeze(2).repeat(1, 1, src_len, 1)
        keys = keys.unsqueeze(1).repeat(1, tgt_len, 1, 1)

        return self.va(F.tanh(self.Wa(torch.cat([query, keys], dim=-1)))).squeeze(-1) # batch_size x tar_len x src_len x 1


class CopyAttention(nn.Module):
    def __init__(self, attn_type, query_dim, key_dim):
        super(CopyAttention, self).__init__()
        if attn_type == "dot":
            assert  query_dim == key_dim
            self.attention = DotAttention(query_dim)
        elif attn_type == "general":
            self.attention = GeneralAttention(query_dim, key_dim)
        elif attn_type == "concat":
            self.attention = ConcatAttention(query_dim, key_dim)
        else:
            raise NotImplementedError

    def forward(self, query, keys, values, mask=None):
        # query: target_len x batch_size x query_dim
        # keys: batch_size x src_len x key_dim
        # values: batch_size x src_len x value_dim
        # mask: batch_size x src_len, mask = 0/-inf

        weight = self.attention._score(query, keys)  # batch_size x target_len x src_len
        if mask is not None:
            weight = weight + mask.unsqueeze(1).expand_as(weight)
        return weight


