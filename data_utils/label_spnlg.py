import torch
import os
from collections import Counter
import numpy as np
import re
from nltk.tokenize import word_tokenize


def get_spnlg_field(tableStr):
    fdict = {}  # (field, pos) -> value
    all_items = re.findall(r',?(.*?)[[](.*?)[]]', tableStr)  # [('name', 'nameVariable'), (' food', 'Chinese food')]
    for item in all_items:  # eatType[pub]
        field, field_value = item
        field = field.strip()
        values = word_tokenize(field_value)
        for i, v in enumerate(values):
            fdict[(field, i)] = v
    return fdict


class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>", "<ent>"]  # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """add extra word, returns idx of word
        :param word: a str
        :param train: bool, if true, then update self.idx2word and w2i; if false, just update w2i
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """add lots of words, assumes train=True
        :param words: a list of words
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, bsz, max_count=50000, add_bos=False, add_eos=False):
        self.dictionary = Dictionary()
        self.value_dict = Dictionary()
        assert 'spnlg' in path.lower()
        self.path = path
        pair_train_src = os.path.join(path, "pair_src.train")
        pair_valid_src = os.path.join(path, "pair_src.valid")
        pair_train_tgt = os.path.join(path, "pair_tgt.train")
        pair_valid_tgt = os.path.join(path, "pair_tgt.valid")
        raw_train_text = os.path.join(path, "raw_tgt.train")
        raw_valid_text = os.path.join(path, "raw_tgt.valid")

        self.gen_vocab = Dictionary()
        self.make_vocab(pair_train_tgt, pair_train_src, raw_train_text, max_count=max_count)
        self.genset.add("<ent>")
        # self.value_dict.add_word("<ent>", train=True)

        # load training data
        pair_sents_train, pair_sk_sents_train, \
            pair_src_feats_train = self.load_paired_data(pair_train_src, pair_train_tgt,
                                                         add_to_dict=True, add_bos=add_bos, add_eos=add_eos)
        raw_sents_train = self.load_raw_data(raw_train_text)

        self.paired_train, _ = self._minibatchify_pair(pair_sents_train, pair_sk_sents_train, pair_src_feats_train, bsz)
        self.raw_train, _ = self._minibatchify_raw(raw_sents_train, bsz)
        del pair_sents_train, pair_sk_sents_train, pair_src_feats_train, raw_sents_train

        # load valid data
        pair_sents_valid, pair_sk_sents_valid, \
            pair_src_feats_valid = self.load_paired_data(pair_valid_src, pair_valid_tgt,
                                                         add_to_dict=False, add_bos=add_bos, add_eos=add_eos)

        self.paired_valid, self.paired_lineno_valid = self._minibatchify_pair(pair_sents_valid, pair_sk_sents_valid,
                                                                              pair_src_feats_valid, bsz)
        self.raw_valid = None
        del pair_sents_valid, pair_sk_sents_valid, pair_src_feats_valid

    def make_vocab(self, pair_tgt, pair_src, raw_tgt, max_count=50000):
        self.word_cnt = Counter()
        genwords, value_vocab = self.get_vocab_from_paired(pair_tgt, pair_src)
        raw_vocab = self.get_vocab_from_raw(raw_tgt) # just to update self.word_cnt
        self.genset = set(genwords.keys())
        tgtkeys = list(self.word_cnt.keys())
        # make sure gen stuff is first
        tgtkeys.sort(key=lambda x: -(x in self.genset))  # in genset first
        voc = tgtkeys[:max_count]
        self.dictionary.bulk_add(voc)
        self.value_dict.bulk_add(list([i for i in value_vocab.keys() if i in voc]))
        self.gen_vocab.bulk_add(list([i for i in genwords.keys() if i in voc]))
        # make sure we did everything right (assuming didn't encounter any special tokens)
        assert self.dictionary.idx2word[5 + len(self.genset) - 1] in self.genset
        assert self.dictionary.idx2word[5 + len(self.genset)] not in self.genset
        self.dictionary.add_word("<ncf1>", train=True)
        self.dictionary.add_word("<ncf2>", train=True)

    def get_vocab_from_paired(self, tgt_path, src_path):
        assert os.path.exists(tgt_path)
        linewords = []
        with open(src_path, 'r') as f:
            for line in f:
                fields = get_spnlg_field(line.strip())  # key, pos -> word
                fieldvals = fields.values()
                self.word_cnt.update(fieldvals)
                linewords.append(set(wrd for wrd in fieldvals))
                self.word_cnt.update([k for k, idx in fields])
                self.word_cnt.update([idx for k, idx in fields])

        genwords = Counter()  # a Counter that records all the vocab in target
        value_words = Counter()
        with open(tgt_path, 'r') as f:
            for l, line in enumerate(f):
                words = word_tokenize(line.strip())
                genwords.update([wrd for wrd in words if wrd not in linewords[l]])
                value_words.update([wrd for wrd in words if wrd in linewords[l]])
                self.word_cnt.update(words)
        return genwords, value_words

    def get_vocab_from_raw(self, path):
        assert os.path.exists(path)
        raw_vocab = Counter()
        with open(path, 'r') as f:
            for l, line in enumerate(f):
                words = word_tokenize(line.strip())
                self.word_cnt.update(words)
                raw_vocab.update(words)
        return raw_vocab

    def get_test_data(self, table_path):
        w2i = self.dictionary.word2idx
        src_feats = []
        original_feats = []
        with open(table_path, 'r') as f:
            for line in f:
                feats = []
                orig = []
                fields = get_spnlg_field(line.strip())  # (key, pos) -> word
                for (key, pos), wrd in fields.items():
                    if key in w2i:
                        featrow = [self.dictionary.add_word(key, False),
                                   self.dictionary.add_word(pos, False),
                                   self.dictionary.add_word(wrd, False)]
                        feats.append(featrow)
                        orig.append((key, pos, wrd))
                src_feats.append(feats)
                original_feats.append(orig)

        src_feat_batches = []
        line_no_tst = []
        for i in range(len(src_feats)):
            # src = torch.LongTensor(src_feats[i]).unsqueeze(0) # 1 x nfield x 3
            # src_feat_batches.append(src)
            src_feat_batches.append(self._pad_srcfeat([src_feats[i]]))
            line_no_tst.append([i])

        return src_feat_batches, original_feats, line_no_tst

    def get_raw_temp(self, raw_fn_in, fn_out=None, num=5, seed=1):
        np.random.seed(seed)  # define random seed for select certain sentence
        with open(raw_fn_in, 'r') as f:
            all_contents = f.read().strip().split('\n')
        select_num = np.random.randint(0, len(all_contents) - 1, (num,))

        if fn_out is not None:
            with open(fn_out, 'w') as fout:
                for i in select_num:
                    fout.write(all_contents[i] + '\n')

        all_raw_tmps = []
        w2i = self.dictionary.word2idx
        for i in select_num:
            line = all_contents[i]
            words = word_tokenize(line.strip(()))
            token = []
            for word in words:
                if word in w2i:
                    token.append(w2i[word])
                else:
                    token.append(w2i['<unk>'])
            # token list to tensor
            token = torch.LongTensor(token)
            all_raw_tmps.append(token)
        return all_raw_tmps

    def get_raw_temp_from_file(self, fn_in):
        all_raw_tmps = []
        with open(fn_in, 'r') as f:
            for line in f:
                words = word_tokenize(line.strip())
                token = []
                for word in words:
                    if word in self.dictionary.word2idx:
                        token.append(self.dictionary.word2idx[word])
                    else:
                        token.append(self.dictionary.word2idx['<unk>'])
                token = torch.LongTensor(token)
                all_raw_tmps.append(token)
        return all_raw_tmps

    def load_paired_data(self, table_path, text_path, add_to_dict=False, add_bos=False, add_eos=False):
        w2i = self.dictionary.word2idx
        sents = []
        sk_sents = []
        raw_sentences = []
        src_feats = []
        linewords = []
        with open(table_path, 'r') as f:
            for line in f:
                fields = get_spnlg_field(line.strip())
                feats = []
                linewords.append(set(fields.values()))
                for (key, pos), wrd in fields.items():
                    if key in w2i:
                        featrow = [self.dictionary.add_word(key, add_to_dict),
                                   self.dictionary.add_word(pos, add_to_dict),
                                   self.dictionary.add_word(wrd, False)]  # value can not update, but key can
                        feats.append(featrow)
                src_feats.append(feats)

        with open(text_path, 'r') as f:
            for l, line in enumerate(f):
                words = word_tokenize(line.strip())
                raw_sentences.append(words)
                token = []
                sk_tokens = []
                if add_bos:
                    token.append(self.dictionary.add_word('<bos>', True))
                    sk_tokens.append(self.dictionary.add_word('<bos>', True))

                for word in words:
                    if word in w2i:
                        token.append(w2i[word])
                    else:
                        token.append(w2i['<unk>'])

                    if word in linewords[l]:
                        if len(sk_tokens) == 0 or (len(sk_tokens) == 1 and sk_tokens[0] == w2i['<bos>']):  # first word
                            sk_tokens.append(self.dictionary.word2idx['<ent>'])
                        elif sk_tokens[-1] == self.dictionary.word2idx['<ent>']:
                            continue
                        else:
                            sk_tokens.append(self.dictionary.word2idx['<ent>'])
                    else:  # ordinary word
                        if word in w2i:
                            sk_tokens.append(w2i[word])
                        else:
                            sk_tokens.append(w2i['<unk>'])
                if add_eos:
                    token.append(self.dictionary.add_word('<eos>', True))
                    sk_tokens.append(self.dictionary.add_word('<eos>', True))
                sents.append(token)
                sk_sents.append(sk_tokens)
        del token, sk_tokens
        assert len(raw_sentences) == len(src_feats)

        return sents, sk_sents, src_feats

    def load_raw_data(self, text_path, add_bos=False, add_eos=False):
        w2i = self.dictionary.word2idx
        sents = []
        with open(text_path, 'r') as f:
            for line in f:
                words = word_tokenize(line.strip())
                token = []
                if add_bos:
                    token.append(self.dictionary.add_word('<bos>', True))
                for word in words:
                    if word in w2i:
                        token.append(w2i[word])
                    else:
                        token.append(w2i['<unk>'])
                if add_eos:
                    token.append(self.dictionary.add_word('<eos>', True))
                sents.append(token)
        return sents

    def _pad_sequence(self, sequence):
        # sequence: b x seq
        max_row = max(len(i) for i in sequence)
        for item in sequence:
            if len(item) < max_row:
                item.extend([self.dictionary.word2idx["<pad>"]] * (max_row - len(item)))
        return torch.LongTensor(sequence)

    def _pad_srcfeat(self, curr_feats):
        # return a  b x nfield(max) x 3
        max_rows = max(len(feats) for feats in curr_feats)
        nfeats = len(curr_feats[0][0])
        for feats in curr_feats:
            if len(feats) < max_rows:
                [feats.append([self.dictionary.word2idx["<pad>"] for _ in range(nfeats)])
                 for _ in range(max_rows - len(feats))]
        return torch.LongTensor(curr_feats)

    # def _pad_loc(self, curr_locs):
    #     """
    #     curr_locs is a bsz-len list of tgt-len list of locations
    #     returns:
    #       a seqlen x bsz x max_locs tensor
    #     """
    #     max_locs = max(len(locs) for blocs in curr_locs for locs in blocs)
    #     max_seq = max(len(blocs) for blocs in curr_locs)
    #     for blocs in curr_locs:
    #         for locs in blocs:
    #             if len(locs) < max_locs:
    #                 locs.extend([-1] * (max_locs - len(locs)))
    #         if len(blocs) < max_seq:
    #             blocs.extend([[-1] * max_locs] * (max_seq - len(blocs)))
    #     return torch.LongTensor(curr_locs).transpose(0, 1).contiguous()

    # def _pad_inp(self, curr_inps):
    #     """
    #     curr_inps is a bsz-len list of seqlen-len list of nlocs-len list of features
    #     returns:
    #       a bsz x seqlen x max_nlocs x nfeats tensor
    #     """
    #     max_locs = max(len(feats) for seq in curr_inps for feats in seq)
    #     max_seq = max(len(seq) for seq in curr_inps)
    #     nfeats = len(curr_inps[0][0][0])  # default: 3
    #     for seq in curr_inps:
    #         for feats in seq:
    #             if len(feats) < max_locs:
    #                 randidxs = [random.randint(0, len(feats) - 1) for _ in
    #                             range(max_locs - len(feats))]  # random from on of feat
    #                 [feats.append(feats[ridx]) for ridx in randidxs]
    #
    #         if len(seq) < max_seq:
    #             seq.extend([seq[-1] * (max_seq - len(seq))])
    #     return torch.LongTensor(curr_inps)

    def _minibatchify_pair(self, sents, sk_sents, src_feats, bsz):

        sents, sorted_idxs = zip(
            *sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))  # from shortest to longest
        minibatches, mb2linenos = [], []
        curr_sent, curr_sk, curr_len, curr_srcfeat = [], [], [], []
        curr_line = []

        for i in range(len(sents)):
            if len(curr_sent) == bsz:  # one batch is done!
                minibatches.append((self._pad_sequence(curr_sent).t().contiguous(),
                                    self._pad_sequence(curr_sk).t().contiguous(),
                                    torch.IntTensor(curr_len),
                                    self._pad_srcfeat(curr_srcfeat)))

                mb2linenos.append(curr_line)
                # init
                curr_line = [sorted_idxs[i]]
                curr_sent, curr_len = [sents[i]], [len(sents[i])]
                curr_sk = [sk_sents[sorted_idxs[i]]]
                curr_srcfeat = [src_feats[sorted_idxs[i]]]

            else:
                curr_sent.append(sents[i])
                curr_len.append(len(sents[i]))
                curr_line.append(sorted_idxs[i])
                curr_sk.append(sk_sents[sorted_idxs[i]])
                curr_srcfeat.append(src_feats[sorted_idxs[i]])

        if len(curr_sent) > 0:  # last
            minibatches.append((self._pad_sequence(curr_sent).t().contiguous(),
                                self._pad_sequence(curr_sk).t().contiguous(),
                                torch.IntTensor(curr_len),
                                self._pad_srcfeat(curr_srcfeat)))
            mb2linenos.append(curr_line)

        return minibatches, mb2linenos

    def _minibatchify_raw(self, sents, bsz):
        sents, sorted_idxs = zip(
            *sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))  # from shortest to longest
        minibatches, mb2linenos = [], []
        curr_sent, curr_len, curr_line = [], [], []
        for i in range(len(sents)):
            if len(curr_sent) == bsz:  # one batch is done!
                minibatches.append((self._pad_sequence(curr_sent).t().contiguous(),
                                    torch.IntTensor(curr_len)))
                mb2linenos.append(curr_line)
                # init
                curr_sent, curr_line, curr_len = [sents[i]], [sorted_idxs[i]], [len(sents[i])]
            else:
                curr_sent.append(sents[i])
                curr_len.append(len(sents[i]))
                # curr_bow.append(sent_bow[sorted_idxs[i]])
                curr_line.append(sorted_idxs[i])

        if len(curr_sent) > 0:  # last
            minibatches.append((self._pad_sequence(curr_sent).t().contiguous(),
                                torch.IntTensor(curr_len)))
            mb2linenos.append(curr_line)

        return minibatches, mb2linenos


if __name__ == "__main__":
    pass
