import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedBackBase(nn.Module):
    def __init__(self, type, lookup):
        super(FeedBackBase, self).__init__()
        self.type = type
        self.lookup = lookup

    def prepare(self, decoder_out):
        raise NotImplementedError

    def collect(self, from_torch=None):
        raise NotImplementedError


class GreedyFeedBack(FeedBackBase):
    def __init__(self, lookup, unk_idx=-1):
        super(GreedyFeedBack, self).__init__(type="greedy", lookup=lookup)
        self.ids = []
        self.unk_idx = unk_idx

    def prepare(self, decoder_out):
        # decoder_out : 1 x b x vocab
        word_prob = F.softmax(decoder_out, dim=-1)
        word_prob[:, :, self.unk_idx].fill_(0) # disvow unk
        return word_prob

    def forward(self, decoder_out, keep_ids=True):
        # decoder_out: 1 x b x vocab
        log_p = self.prepare(decoder_out)
        # log_p: [1, batch_size, vocab_size]
        max_id = torch.argmax(log_p[0], dim=-1).unsqueeze(0) # [1 x batch_size]
        if keep_ids:
            self.ids.append(max_id)
        return max_id

    def clear_ids(self):
        self.ids = []

    def collect(self, from_torch=None):
        # from_torch:
        if from_torch is None:
            ret = torch.cat(self.ids, dim=0)  # [seq_len, batch_size, kmul]
            self.ids = []
            return ret
        else:
            return torch.argmax(from_torch, dim=-1)


class SampleFeedBack(FeedBackBase):
    def __init__(self, lookup, unk_idx=-1):
        super(SampleFeedBack, self).__init__(type="sample", lookup=lookup)
        self.ids = []
        self.unk_idx = unk_idx

    def prepare(self, decoder_out):
        # decoder_out : 1 x b x vocab
        word_prob = F.softmax(decoder_out, dim=-1)
        word_prob[:, :, self.unk_idx].fill_(0)  # disenable unk
        return word_prob

    def forward(self, decoder_out):
        log_p = self.prepare(decoder_out)
        # log_p: [1, batch_size*beam_size, vocab_size]
        sample_idx = torch.multinomial(log_p[0], 1) # b x 1
        self.ids.append(sample_idx.transpose(0, 1)) # append 1 x b
        return sample_idx
        # return self.lookup(sample_idx)

    def clear_ids(self):
        self.ids = []

    def collect(self):
        ret = torch.cat(self.ids, dim=0) # seq x b
        self.ids = []
        return ret

class SampleFeedBackWithTemperature(FeedBackBase):
    def __init__(self, lookup, unk_idx=-1, temperature=1.0):
        super(SampleFeedBackWithTemperature, self).__init__(type="temperature_sample", lookup=lookup)
        self.ids = []
        self.unk_idx = unk_idx
        self.temperature = temperature

    def prepare(self, decoder_out):
        word_prob = F.softmax(decoder_out/self.temperature, dim=-1)
        word_prob[:, :, self.unk_idx].fill_(0)
        return word_prob

    def forward(self, decoder_out):
        log_p = self.prepare(decoder_out)
        # log_p: [1, batch_size, vocab_size]
        sample_idx = torch.multinomial(log_p[0], 1)  # b x 1
        self.ids.append(sample_idx.transpose(0, 1))  # append 1 x b
        return sample_idx

    def clear_ids(self):
        self.ids = []

    def collect(self):
        ret = torch.cat(self.ids, dim=0)
        self.ids = []
        return ret

class TopkSampleFeedBack(FeedBackBase):
    def __init__(self, lookup, unk_id=-1, topk=1):
        super(TopkSampleFeedBack, self).__init__(type="topk_sample", lookup=lookup)
        self.ids = []
        self.unk_idx = unk_id
        self.topk = topk

    def prepare(self, decoder_out):
        # 1 x b x vocab
        idx_to_remove = decoder_out < torch.topk(decoder_out, self.topk)[0][..., -1, None]
        decoder_out[idx_to_remove] = -float("inf")
        word_prob = F.softmax(decoder_out, dim=-1)
        word_prob[:, :, self.unk_idx].fill_(0)
        return word_prob

    def forward(self, decoder_out):
        self.topk = min(self.topk, decoder_out.size(-1)) # safety check
        log_p = self.prepare(decoder_out) # 1 x b x vocab
        sample_idx = torch.multinomial(log_p[0], 1)  # b x 1
        self.ids.append(sample_idx.transpose(0, 1))  # append 1 x b
        return sample_idx

    def clear_ids(self):
        self.ids = []

    def collect(self):
        ret = torch.cat(self.ids, dim=0)
        self.ids = []
        return ret


class NucleusSampleFeedBack(FeedBackBase):
    def __init__(self, lookup, unk_id=-1, topp=1.0):
        super(NucleusSampleFeedBack, self).__init__(type="nucleus_sample", lookup=lookup)
        self.ids = []
        self.unk_idx = unk_id
        self.topp = topp

    def prepare(self, decoder_out):
        sorted_value, sorted_idx = torch.sort(decoder_out, descending=True)
        cumulated_p = torch.cumsum(F.softmax(sorted_value, dim=-1), dim=-1)
        sort_idx_to_remove = cumulated_p > self.topp
        # assure there must be one element in each batch
        sort_idx_to_remove[..., 1:] = sort_idx_to_remove[..., :-1].clone()
        sort_idx_to_remove[..., 0] = 0
        id_to_remove = torch.gather(sort_idx_to_remove, -1, sorted_idx)
        decoder_out[id_to_remove] = -float("inf")
        word_prob = F.softmax(decoder_out, dim=-1)
        word_prob[:, :, self.unk_idx].fill_(0)
        return word_prob

    def forward(self, decoder_out):
        log_p = self.prepare(decoder_out) # 1 x b x vocab
        sample_idx = torch.multinomial(log_p[0], 1)
        self.ids.append(sample_idx.transpose(0, 1))
        return sample_idx

    def clear_ids(self):
        self.ids = []

    def collect(self):
        ret = torch.cat(self.ids, dim=0)
        self.ids = []
        return ret

class BeamFeedBack(FeedBackBase):
    """
    a helper class for inferenece with beam search
    """
    def __init__(self, lookup, beam_size, unk_idx=-1):
        super(BeamFeedBack, self).__init__(type="beam", lookup=lookup)
        self.beam_size = beam_size
        self.output_size = lookup.num_embeddings
        self.unk_idx = unk_idx
        self.back_pointers = []
        self.symbols = []

    def repeat(self, v):
        # v: batch_size x ?
        return v.repeat(1, self.beam_size, 1) # v: 1x bx voc
        # return v.unsqueeze(1).repeat(1, self.beam_size, 1) # 1 x b*beam x ?

    def forward(self, past_p, cur_p, batch_size, step, keep_ids=True):
        # cur_p: [batch*beam, vocab_size]
        # past_p: [batch*beam, 1]
        if step == 0:
            score = cur_p.view(batch_size, -1)[:, 0:self.output_size]
        else:
            score = (cur_p + past_p).view(batch_size, -1)
        top_v, top_id = score.topk(self.beam_size, dim=1)

        back_ptr = top_id.div(self.output_size) # which beam
        symbols = top_id.fmod(self.output_size) # which word
        past_p = top_v.view(-1, 1)
        if keep_ids:
            self.back_pointers.append(back_ptr.view(-1, 1))
            self.symbols.append(symbols.view(-1, 1))

        return past_p, symbols

    def clear_ids(self):
        self.symbols = []
        self.back_pointers = []

    def collect(self, past_p, batch_size):
        # past_p: b*beam x 1
        final_seq_symbols = []
        cum_sum = past_p.view(-1, self.beam_size) # b x beam

        max_seq_ids = cum_sum.topk(self.beam_size)[1] # batch_size x beam_size # .data.cpu().view(-1).numpy()

        rev_seq_symbols = self.symbols[::-1]
        rev_back_ptrs = self.back_pointers[::-1]

        for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
            symbol2ds = symbols.view(-1, self.beam_size)
            back2ds = back_ptrs.view(-1, self.beam_size)

            selected_symbols = []
            selected_parents = []
            for b_id in range(batch_size):
                selected_parents.append(back2ds[b_id, max_seq_ids[b_id]])
                selected_symbols.append(symbol2ds[b_id, max_seq_ids[b_id]])
                # print(back2ds[b_id, max_seq_ids[b_id]])
                # print(symbol2ds[b_id, max_seq_ids[b_id]])

            final_seq_symbols.append(torch.stack(selected_symbols).unsqueeze(1))
            max_seq_ids = torch.stack(selected_parents)# .data.cpu().numpy()

        sequence_symbols = torch.cat(final_seq_symbols[::-1], dim=1) # batch_size x seq_len x beam_size
        sequence_symbols = sequence_symbols[:, :, 0].transpose(0, 1)
        return sequence_symbols


if __name__ == "__main__":
    vocab = 100
    seq = 5
    emb = 30
    pad_idx = 1
    batch = 4
    beam_size = 3
    lut = nn.Embedding(vocab, emb, padding_idx=pad_idx)
    print(lut.num_embeddings)
    # print("use greedy")
    feedback = GreedyFeedBack(lut, unk_idx=0)
    beam_fb = BeamFeedBack(lut, beam_size)
    past_p = torch.zeros(batch * beam_size, 1)

    for t in range(seq):
        word_dis = torch.randn(1, batch, vocab)
        # max_ids = feedback(word_dis.squeeze(0))
        max_ids = feedback(word_dis)[0][0].item()
        print("greedy - max_ids", max_ids)

        # print("greedy - max_ids", max_ids.tolist())
        cur_p = beam_fb.repeat(word_dis).squeeze(0)
        # print("cur_p size", cur_p.size())
        past_p, symbol = beam_fb(past_p, cur_p, batch, t)
        print(symbol[0][0].item())
        print("beam - most possible symbol:", symbol[:, 0].tolist())
        # print(beam_fb.back_pointers)

    greedy_ids = feedback.collect() # seq x b
    # print(sentences_ids)
    print(greedy_ids.size())

    beam_ids = beam_fb.collect(past_p, batch)
    print(beam_ids)
