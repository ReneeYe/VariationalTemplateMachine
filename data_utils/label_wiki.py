import torch
import os
from collections import Counter
import numpy as np


def get_wiki_poswrds(tokes):
    """(key, num) -> word"""
    fields = {}
    for toke in tokes:
        try:
            fullkey, val = toke.split(':')
        except ValueError:
            ugh = toke.split(':') # must be colons in the val
            fullkey = ugh[0]
            val = ''.join(ugh[1:])
        if val == "<none>":
            continue
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
            keynum = 1
        else:
            keynum = int(keypieces[-1])
            key = '_'.join(keypieces[:-1])
        fields[key, keynum] = val
    return fields

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
        self.path = path
        pair_train_src = os.path.join(path, "pair_src.train")
        pair_valid_src = os.path.join(path, "pair_src.valid")
        pair_train_tgt = os.path.join(path, "pair_tgt.train")
        pair_valid_tgt = os.path.join(path, "pair_tgt.valid")

        # if not ner_fake:
        #     pair_train_src = os.path.join(path, "pair_src_train.txt")
        #     pair_valid_src = os.path.join(path, "pair_src_valid.txt")
        #     pair_train_text = os.path.join(path, "pair_train.txt")
        #     pair_valid_text = os.path.join(path, "pair_valid.txt")
        # else:
        #     pair_train_src = os.path.join(path, "pair_src_train_all_include.txt")
        #     pair_valid_src = os.path.join(path, "pair_src_valid.txt")
        #     pair_train_text = os.path.join(path, "pair_train_all_include.txt")
        #     pair_valid_text = os.path.join(path, "pair_valid.txt")

        raw_train_text = os.path.join(path, "raw_tgt.train")
        raw_valid_text = os.path.join(path, "raw_tgt.valid")

        self.gen_vocab = Dictionary()
        self.make_vocab(pair_train_tgt, pair_train_src, raw_train_text, max_count=max_count)
        self.genset.add("<ent>")

        # load training data
        pair_sents_train, pair_sk_sents_train, \
            pair_src_feats_train = self.load_paired_data(pair_train_src, pair_train_tgt,
                                                         add_to_dict=False, add_bos=add_bos, add_eos=add_eos)
        raw_sents_train = self.load_raw_data(raw_train_text)

        self.paired_train, _ = self._minibatchify_pair(pair_sents_train, pair_sk_sents_train, pair_src_feats_train, bsz)
        self.raw_train, _ = self._minibatchify_raw(raw_sents_train, bsz)
        del pair_sents_train, pair_sk_sents_train, pair_src_feats_train, raw_sents_train

        # load valid data
        pair_sents_valid, pair_sk_sents_valid,  \
            pair_src_feats_valid = self.load_paired_data(pair_valid_src, pair_valid_tgt,
                                                         add_to_dict=False, add_bos=add_bos, add_eos=add_eos)
        raw_sents_valid = self.load_raw_data(raw_valid_text)
        self.paired_valid, self.paired_lineno_valid = self._minibatchify_pair(pair_sents_valid, pair_sk_sents_valid,
                                                                              pair_src_feats_valid, bsz)
        if len(raw_sents_valid) == 0:
            self.raw_valid = None
        else:
            self.raw_valid, _ = self._minibatchify_raw(raw_sents_valid, bsz)

        del pair_sents_valid, pair_sk_sents_valid, pair_src_feats_valid, raw_sents_valid

    def make_vocab(self, pair_tgt, pair_src, raw_tgt, max_count=50000):
        self.word_cnt = Counter()
        genwords, value_vocab = self.get_vocab_from_paired(pair_tgt, pair_src)
        raw_vocab = self.get_vocab_from_raw(raw_tgt) # just to update self.word_cnt
        self.genset = set(genwords.keys())
        tgtkeys = list(self.word_cnt.keys())
        tgtkeys.sort(key=lambda x: -(x in self.genset)) # add genset first
        voc = tgtkeys[:max_count]
        self.dictionary.bulk_add(voc)
        self.value_dict.bulk_add(list([i for i in value_vocab.keys() if i in voc]))
        self.gen_vocab.bulk_add(list([i for i in genwords.keys() if i in voc]))
        # make sure we did everything right (assuming didn't encounter any special tokens)
        assert self.dictionary.idx2word[5 + len(self.genset) - 1] in self.genset
        assert self.dictionary.idx2word[5 + len(self.genset)] not in self.genset
        self.dictionary.add_word("<ncf1>", train=True)
        self.dictionary.add_word("<ncf2>", train=True)

    def get_vocab_from_paired(self, path, src_path):
        assert os.path.exists(path)
        linewords = []
        with open(src_path, 'r') as f:
            for line in f:
                tokes = line.strip().split()
                fields = get_wiki_poswrds(tokes) # key, pos -> wrd
                fieldvals = fields.values()
                self.word_cnt.update(fieldvals)
                linewords.append(set(wrd for wrd in fieldvals))
                self.word_cnt.update([k for k, idx in fields])
                self.word_cnt.update([idx for k, idx in fields])

        genwords = Counter()  # a Counter that records all the vocab in target
        value_words = Counter()
        with open(path, 'r') as f:
            for l, line in enumerate(f):
                words, spanlabels = line.strip().split('|||')
                words = words.split()
                genwords.update([wrd for wrd in words if wrd not in linewords[l]])
                value_words.update([wrd for wrd in words if wrd in linewords[l]])
                self.word_cnt.update(words)

        genwords = {k: v for k,v in genwords.items() if v > 5}

        return genwords, value_words

    def get_vocab_from_raw(self, path):
        assert os.path.exists(path)
        raw_vocab = Counter()
        with open(path, 'r') as f:
            for l, line in enumerate(f):
                words = line.strip().split()
                self.word_cnt.update(words)
                raw_vocab.update(words)
        return raw_vocab

        # sent_voc = {k: v for k, v in wrdcnt.items() if v > thresh}
        # # rare_words = {k: v for k, v in wrdcnt.items() if v > value_thresh1 and v<10} # set rare words as value words
        # # rare_words = {k:v for k,v in wrdcnt.items() if v==1}
        # self.genset.update(set(sent_voc.keys()))
        # self.gen_vocab.bulk_add(list(sent_voc.keys()))
        # self.dictionary.bulk_add(list(sent_voc.keys()))
        # # self.value_dict.bulk_add(list(rare_words.keys()))
        # del sent_voc

    def get_test_data(self, table_path):
        w2i = self.dictionary.word2idx
        src_feats = []
        original_feats = []
        with open(table_path, 'r') as f:
            for line in f:
                feats = []
                orig = []
                items = line.strip().split()
                fields = get_wiki_poswrds(items) # (key, pos) -> wrd
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
        np.random.seed(seed) # define random seed for select certain sentence
        with open(raw_fn_in,'r') as f:
            all_contents = f.read().strip().split('\n')
        select_num = np.random.randint(0, len(all_contents)-1, (num,))

        if fn_out is not None:
            with open(fn_out, 'w') as fout:
                for i in select_num:
                    if '|||' in all_contents[i]:
                        words, labels = all_contents[i].strip().split('|||')
                        fout.write(words + '\n')
                    else:
                        fout.write(all_contents[i] + '\n')

        all_raw_tmps = []
        w2i = self.dictionary.word2idx
        for i in select_num:
            line = all_contents[i]
            if '|||' in line: # for paired data
                words, labels = line.strip().split('|||')
                words = words.split()
            else: # for raw data
                words = line.strip().split()
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
                words = line.strip().split()
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
                items = line.strip().split()
                fields = get_wiki_poswrds(items)  # dict: (key, pos) -> word
                feats = []
                linewords.append(set(fields.values()))
                for (key, pos), wrd in fields.items():
                    if key in w2i:
                        featrow = [self.dictionary.add_word(key, add_to_dict),
                                   self.dictionary.add_word(pos, add_to_dict),
                                   self.dictionary.add_word(wrd, False)]  # word can not update, but key can
                        feats.append(featrow)
                src_feats.append(feats)

        with open(text_path, 'r') as f:
            for l, line in enumerate(f):
                words, labels = line.strip().split('|||')
                words = words.split()
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
                        if len(sk_tokens) == 0 or (len(sk_tokens) == 1 and sk_tokens[0] == w2i['<bos>']): # first word
                            sk_tokens.append(self.dictionary.word2idx['<ent>'])
                        elif sk_tokens[-1] == self.dictionary.word2idx['<ent>']:
                            continue
                        else:
                            sk_tokens.append(self.dictionary.word2idx['<ent>'])
                    else: # ordinary word
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
                words = line.strip().split()
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

    def _pad_sequence(self, sequence, ner=False):
        # sequence: b x seq
        max_row = max(len(i) for i in sequence)
        for item in sequence:
            if len(item) < max_row:
                if ner:
                    item.extend([self.tag_dict.word2idx['<pad>']]*(max_row-len(item)))
                else:
                    item.extend([self.dictionary.word2idx["<pad>"]]*(max_row-len(item)))
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
    # table = "name_1:walter	name_2:extra	image:<none>	image_size:<none>	caption:<none>	birth_name:<none>	birth_date_1:1954	birth_place:<none>	death_date:<none>	death_place:<none>	death_cause:<none>	resting_place:<none>	resting_place_coordinates:<none>	residence:<none>	nationality_1:german	ethnicity:<none>	citizenship:<none>	other_names:<none>	known_for:<none>	education:<none>	alma_mater:<none>	employer:<none>	occupation_1:aircraft	occupation_2:designer	occupation_3:and	occupation_4:manufacturer	home_town:<none>	title:<none>	salary:<none>	networth:<none>	height:<none>	weight:<none>	term:<none>	predecessor:<none>	successor:<none>	party:<none>	boards:<none>	religion:<none>	spouse:<none>	partner:<none>	children:<none>	parents:<none>	relations:<none>	signature:<none>	website:<none>	footnotes:<none>	article_title_1:walter	article_title_2:extra"
    # sentence = "walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra flugzeugbau -lrb- extra aircraft construction -rrb- , a manufacturer of aerobatic aircraft . <eos>"

    # table = "image:<none>	name_1:carlene	name_2:m.	name_3:walker	caption:<none>	state_senate_1:utah	term_start_1:january	term_start_2:15	term_start_3:,	term_start_4:2001	term_end_1:present	predecessor_1:scott	predecessor_2:n.	predecessor_3:howell	successor_1:karen	successor_2:morgan	district_1:8th	birth_date_1:2	birth_date_2:september	birth_date_3:1947	birth_place_1:san	birth_place_2:francisco	birth_place_3:,	birth_place_4:ca	party_1:republican	party_2:party	spouse_1:gordon	residence_1:salt	residence_2:lake	residence_3:city	residence_4:,	residence_5:ut	occupation_1:businesswoman	religion_1:latter-day	religion_2:saint	website_1:-lsb-	website_2:http://www.utahsenate.org/perl/spage/distbio2007.pl?dist8	website_3:legislative	website_4:website	website_5:-rsb-	article_title_1:carlene	article_title_2:m.	article_title_3:walker"
    # sentence = "carlene m. walker is an american politician and businesswoman from utah . <eos>"

    # table = "name_1:bundit	name_2:ungrangsee	image_1:bundit	image_2:ungrangsee	image_3:19072007	image_4:bkkiff.jpg	image_size_1:200px	caption_1:bundit	caption_2:attends	caption_3:the	caption_4:opening	caption_5:party	caption_6:for	caption_7:the	caption_8:2007	caption_9:bangkok	caption_10:international	caption_11:film	caption_12:festival	caption_13:.	birth_date_1:7	birth_date_2:december	birth_date_3:1970	birth_place_1:thailand	death_date:<none>	death_place:<none>	education_1:university	education_2:of	education_3:michigan	occupation_1:conductor	title:<none>	spouse_1:mary	spouse_2:ungrangsee	parents:<none>	children:<none>	nationality_1:thai	website_1:-lsb-	website_2:http://www.bundit.org	website_3:www.bundit.org	website_4:-rsb-	article_title_1:bundit	article_title_2:ungrangsee"
    # sentence = "bundit ungrangsee -lrb- ; , born december 7 , 1970 -rrb- is an international symphonic conductor . <eos>"

    # table = "name_1:jole	name_2:fierro	image_1:jole	image_2:fierro.jpg	image_size:<none>	caption:<none>	birth_name:<none>	birth_date_1:22	birth_date_2:november	birth_date_3:1926	birth_place_1:salerno	birth_place_2:,	birth_place_3:italy	death_date_1:27	death_date_2:march	death_date_3:1988	death_place_1:rome	death_place_2:,	death_place_3:italy	occupation_1:actress	spouse:<none>	article_title_1:jole	article_title_2:fierro"
    # sentence = "jole fierro -lrb- 22 november 1926 - 27 march 1988 -rrb- was an italian actress . <eos>"
    #
    # table = "name_1:roberto	name_2:la	name_3:rocca	image:<none>	imagesize:<none>	caption:<none>	nationality_1:ven	nationality_2:venezuelan	birth_date_1:09	birth_date_2:july	birth_date_3:1992	birth_place_1:caracas	birth_place_2:,	birth_place_3:venezuela	starts:<none>	wins:<none>	poles:<none>	year:<none>	titles_1:f2000	titles_2:championship	titles_3:series	article_title_1:roberto	article_title_2:la	article_title_3:rocca"
    # sentence = "roberto la rocca -lrb- born 9 july 1991 -rrb- is a venezuelan racing driver . <eos>"
    # table = "name_1:george	name_2:ratcliffe	image:<none>	country_1:england	fullname_1:george	fullname_2:ratcliff	fullname_3:-lrb-	fullname_4:e	fullname_5:-rrb-	height:<none>	nickname:<none>	birth_date_1:1	birth_date_2:april	birth_date_3:1856	birth_place_1:ilkeston	birth_place_2:,	birth_place_3:derbyshire	birth_place_4:,	birth_place_5:england	death_date_1:7	death_date_2:march	death_date_3:1928	death_place_1:nottingham	death_place_2:,	death_place_3:england	batting_1:left-handed	batting_2:batsman	bowling:<none>	role:<none>	family:<none>	international:<none>	testdebutdate:<none>	testdebutyear:<none>	testdebutagainst:<none>	testcap:<none>	lasttestdate:<none>	lasttestyear:<none>	lasttestagainst:<none>	club_1:derbyshire	year_1:1887	year_2:&	year_3:ndash	year_4:;	year_5:1889	type_1:first-class	debutdate_1:27	debutdate_2:june	debutyear_1:1887	debutfor_1:derbyshire	debutagainst_1:yorkshire	lastdate_1:15	lastdate_2:august	lastyear_1:1887	lastfor_1:derbyshire	lastagainst_1:surrey	deliveries_1:balls	deliveries_2:12	columns_1:1	column_1:first-class	matches_1:5	runs_1:145	s/s_1:0/1	wickets_1:0	fivefor:<none>	tenfor:<none>	catches/stumpings_1:0	catches/stumpings_2:/	catches/stumpings_3:-	date_1:1856	date_2:4	date_3:1	date_4:yes	source_1:http://cricketarchive.com/players/32/32252/32252.html	article_title_1:george	article_title_2:ratcliffe	article_title_3:-lrb-	article_title_4:cricketer	article_title_5:,	article_title_6:born	article_title_7:1856	article_title_8:-rrb-"
    # sentence = "`` another cricketer who played for derbyshire during the 1919 season was named george ratcliffe . '' <eos>"

    table = "name_1:paul	name_2:of	name_3:thebes	birth_date_1:c.	birth_date_2:227	birth_date_3:ad	death_date_1:c.	death_date_2:342	death_date_3:ad	feast_day_1:february	feast_day_2:9	feast_day_3:-lrb-	feast_day_4:oriental	feast_day_5:orthodox	feast_day_6:churches	feast_day_7:-rrb-	group_1:note	group_2:-rcb-	group_3:-rcb-	venerated_in_1:oriental	venerated_in_2:orthodox	venerated_in_3:churches	venerated_in_4:catholic	venerated_in_5:church	venerated_in_6:eastern	venerated_in_7:orthodox	venerated_in_8:church	image_1:anba_bola_1	image_2:.	image_3:gif	caption_1:''	caption_2:saint	caption_3:paul	caption_4:,	caption_5:`	caption_6:the	caption_7:first	caption_8:hermit	caption_9:'	caption_10:''	birth_place_1:egypt	death_place_1:monastery	death_place_2:of	death_place_3:saint	death_place_4:paul	death_place_5:the	death_place_6:anchorite	death_place_7:,	death_place_8:egypt	titles_1:the	titles_2:first	titles_3:hermit	beatified_date:<none>	beatified_place:<none>	beatified_by:<none>	canonized_date:<none>	canonized_place:<none>	canonized_by:<none>	attributes_1:two	attributes_2:lions	attributes_3:,	attributes_4:palm	attributes_5:tree	attributes_6:,	attributes_7:raven	patronage:<none>	major_shrine_1:monastery	major_shrine_2:of	major_shrine_3:saint	major_shrine_4:paul	major_shrine_5:the	major_shrine_6:anchorite	major_shrine_7:,	major_shrine_8:egypt	suppressed_date:<none>	issues:<none>	article_title_1:paul	article_title_2:of	article_title_3:thebes"
    sentence = "paul of thebes , commonly known as paul , the first hermit or paul the anchorite -lrb- d. c. 341 -rrb- is regarded as the first christian hermit . <eos>"

    table = table.strip().split()
    fields = get_wiki_poswrds(table) # key, pos -> wrd
    line_words = set(fields.values())
    words = sentence.strip().split()
    sk_words = []
    for word in words:
        if word in line_words:
            if len(sk_words) == 0:
                sk_words.append('<ent>')
            elif sk_words[-1] == '<ent>':
                continue
            else:
                sk_words.append('<ent>')
        else:
            sk_words.append(word)
    print("skeleton: {}".format(" ".join(sk_words)))
