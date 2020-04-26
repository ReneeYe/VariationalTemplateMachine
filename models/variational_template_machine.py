import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import feedback
from models import attention
import numpy as np


def logsumexp1(X):
    """ X - B x K
    returns: B x 1
    """
    maxes, _ = torch.max(X, 1, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X - maxes.expand_as(X)), 1, True))
    return lse


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar) -
                           torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar)) -
                           torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5*torch.sum(logvar + np.log(2*np.pi) + torch.div(torch.pow((x-mu), 2), torch.exp(logvar)), 1)


def sample_from_gaussian(mu, logvar, seed=None):
    if seed is None:
        epsilon = logvar.new_empty(logvar.size()).normal_() # train
    else: # during generation set different random seed
        if not logvar.is_cuda: # if it is not on gpu
            epsilon = logvar.new_empty(logvar.size()).normal_(generator=torch.manual_seed(seed))
        else: # if tensor is on gpu
            epsilon = logvar.new_empty(logvar.size()).normal_(generator=torch.cuda.manual_seed_all(seed))
    std = torch.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z


class VariationalTemplateMachine(nn.Module):
    def __init__(self, corpus, opt):
        super(VariationalTemplateMachine, self).__init__()
        self.use_cuda = opt.cuda
        self.corpus = corpus
        self.unk_idx = corpus.dictionary.word2idx[corpus.dictionary.unk_word]
        self.pad_idx = opt.pad_idx
        self.ent_idx = corpus.dictionary.word2idx["<ent>"]
        self.vocab_size = len(corpus.dictionary)
        self.gen_vocab_size = len(corpus.gen_vocab) # skelton vocab
        self.drop_emb = opt.drop_emb
        self.value_dict_size = len(corpus.value_dict)
        # self.w2i = corpus.dictionary.word2idx

        self.max_seqlen = opt.max_seqlen
        self.initrange = opt.initrange
        self.inp_feats = 3 # key, pos, value
        self.emb_size = opt.emb_size
        self.drop = nn.Dropout(opt.dropout)

        # encode table
        self.table_hid_size = opt.table_hid_size
        self.word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.pad_idx)
        self.table_hidden_out = nn.Linear(self.emb_size * self.inp_feats, self.table_hid_size, bias=True)
        self.pool_type = opt.pool_type

        self.zeros = torch.zeros(1, 1)
        if opt.cuda:
            self.zeros = self.zeros.cuda()

        # encoder sentence x
        # self.birnn = opt.birnn
        self.hid_size = opt.hid_size
        self.layers = opt.layers
        self.rnn_encode = nn.GRU(self.emb_size, self.hid_size, self.layers, dropout=opt.dropout, bidirectional=True)
        self.sent_represent = opt.sent_represent

        # prior net: p(z) follows N(0,I)
        self.z_latent_size = opt.z_latent_size
        self.c_latent_size = self.table_hid_size # latent size of c should be equal to table hidden size

        # posterior net(inference): q(z|x)
        inf_input_size = opt.layers * opt.hid_size * 2 # b x (2*emb+layer*hid)
        self.z_posterior = nn.Linear(inf_input_size, self.z_latent_size * 2)  # mu_z & logvar_z
        self.c_posterior = nn.Linear(inf_input_size, self.c_latent_size * 2)  # mu_c & logvat_c

        # generator p(X|z,h_tab)
        self.z2dec = nn.Linear(self.z_latent_size + self.table_hid_size, self.hid_size)

        self.word_rnn = nn.LSTM(self.z_latent_size + self.emb_size + self.table_hid_size,
                                self.hid_size, self.layers, dropout=opt.dropout)
        self.use_dec_attention = opt.dec_attention
        if self.use_dec_attention:
            self.attn_table_hidden = attention.GeneralAttention(query_dim=self.hid_size, key_dim=self.table_hid_size)
            self.generator_out = nn.Linear(opt.hid_size + self.table_hid_size, self.vocab_size + 1)
        else:
            self.generator_out = nn.Linear(opt.hid_size, self.vocab_size + 1)

        self.set_decode_method(opt)

        # setup loss weights etc
        # self.add_vbow = opt.add_vbow
        # if self.add_vbow:
        #     self.value_bow_z_proj = nn.Linear(self.z_latent_size, self.value_dict_size) # max H
        #     self.value_bow_c_proj = nn.Linear(self.c_latent_size, self.value_dict_size) # min CE
        #     self.vbow_weight_z = opt.vbow_weight_z
        #     self.vbow_weight_c = opt.vbow_weight_c

        self.add_mi_z = opt.add_mi_z
        self.mi_z_weight = opt.mi_z_weight
        self.add_mi_c = opt.add_mi_c
        self.mi_c_weight = opt.mi_c_weight

        # self.add_genbow = opt.add_genbow
        # if self.add_genbow:
        #     self.gen_bow_z_proj = nn.Linear(self.z_latent_size, self.gen_vocab_size) # min CE
        #     self.gen_bow_c_proj = nn.Linear(self.c_latent_size, self.gen_vocab_size) # max H
        #     self.genbow_weight_z = opt.genbow_weight_z
        #     self.genbow_weight_c = opt.genbow_weight_c

        self.add_preserving_template_loss = opt.add_preserving_template_loss
        if opt.add_preserving_template_loss:
            self.pt_weight = opt.pt_weight
            self.template_rnn = nn.LSTM(self.z_latent_size + self.emb_size, self.hid_size, self.layers, dropout=opt.dropout)
            self.template_out = nn.Linear(self.hid_size, self.vocab_size+1)

        # self.add_skeleton = opt.add_skeleton
        # if self.add_skeleton:
        #     self.sk_weight = opt.sk_weight
        #     # decoder for x_skeleton
        #     self.sk_rnn = nn.LSTM(self.z_latent_size + self.emb_size, self.hid_size, self.layers, dropout=opt.dropout)
        #     self.sk_generator_out = nn.Linear(self.hid_size, self.vocab_size + 1)

        # setup annealing tricks for latent variable
        self.anneal_function_z = opt.anneal_function_z
        self.anneal_k_z = opt.anneal_k_z
        self.anneal_x0_z = opt.anneal_x0_z
        self.kl_weight_z = 0.5

        self.anneal_function_c = opt.anneal_function_c
        self.anneal_k_c = opt.anneal_k_c
        self.anneal_x0_c = opt.anneal_x0_c
        self.kl_weight_c = 0.5

        self.add_preserving_content_loss = opt.add_preserving_content_loss
        self.pc_weight = opt.pc_weight
        self.rawloss_weight = opt.rawloss_weight

        self.init_weight()

    def init_weight(self):
        initrange = self.initrange
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.word_emb.weight.data[self.pad_idx].zero_()
        self.word_emb.weight.data[self.corpus.dictionary.word2idx["<ncf1>"]].zero_()
        self.word_emb.weight.data[self.corpus.dictionary.word2idx["<ncf2>"]].zero_()
        self.word_emb.weight.data[self.corpus.dictionary.word2idx['<ent>']].zero_()

        # init generator rnn
        for thing in self.word_rnn.parameters():
            thing.data.uniform_(-initrange, initrange)
        for thing in self.rnn_encode.parameters():
            thing.data.uniform_(-initrange, initrange)

    def set_kl_weight(self, step):
        # set kl weight for z
        if self.anneal_function_z == 'logistic':
            self.kl_weight_z = float(1 / (1 + np.exp(-self.anneal_k_z * (step - self.anneal_x0_z))))
        elif self.anneal_function_z == 'linear':
            self.kl_weight_z = min(1, step / self.anneal_x0_z)
        elif self.anneal_function_z == 'const':
            self.kl_weight_z = self.anneal_k_z
        else:
            self.kl_weight_z = 0.5

        # set kl weight of c
        if self.anneal_function_c == 'logistic':
            self.kl_weight_c = float(1 / (1 + np.exp(-self.anneal_k_c * (step - self.anneal_x0_c))))
        elif self.anneal_function_c == 'linear':
            self.kl_weight_c = min(1, step / self.anneal_x0_c)
        elif self.anneal_function_z == 'const':
            self.kl_weight_c = self.anneal_k_c
        else:
            self.kl_weight_c = 0.5

    def set_decode_method(self, opt):
        self.beamsz = 1
        self.decode_method = opt.decode_method.lower()
        if "beam_search" in self.decode_method:
            self.beamsz = opt.beamsz
            if self.beamsz == 1:
                self.feedback_x = feedback.GreedyFeedBack(self.word_emb, self.unk_idx)
            else:
                self.feedback_x = feedback.BeamFeedBack(self.word_emb, self.beamsz, self.unk_idx)
        elif self.decode_method == "temp_sample":
            self.sample_temperature = opt.sample_temperature
            if self.sample_temperature == 1:
                self.feedback_x = feedback.SampleFeedBack(self.word_emb, self.unk_idx)
            elif self.sample_temperature < 0.001:  # if too small, then it is a greedy method
                self.feedback_x = feedback.GreedyFeedBack(self.word_emb, self.unk_idx)
            else:
                self.feedback_x = feedback.SampleFeedBackWithTemperature(self.word_emb, self.unk_idx,
                                                                         temperature=self.sample_temperature)
        elif self.decode_method == "topk_sample":
            self.topk = opt.topk
            if self.topk == 1:
                self.feedback_x = feedback.GreedyFeedBack(self.word_emb, self.unk_idx)
            else:
                self.feedback_x = feedback.TopkSampleFeedBack(self.word_emb, self.unk_idx, self.topk)

        elif self.decode_method == "nucleus_sample":
            self.topp = opt.topp
            if self.topp == 1:
                self.feedback_x = feedback.SampleFeedBack(self.word_emb, self.unk_idx)
            else:
                self.feedback_x = feedback.NucleusSampleFeedBack(self.word_emb, self.unk_idx, self.topp)

        else:
            self.feedback_x = feedback.GreedyFeedBack(self.word_emb, self.unk_idx)

    def _get_posterior_input(self, y_out, h_yt):
        bsz = h_yt.size(1)
        if self.sent_represent == "last_hid":
            h_yt = torch.transpose(h_yt, 0, 1).contiguous().view(bsz, -1)  # b x layer*hid
            posterior_input = h_yt
        elif self.sent_represent == "seq_avg":
            y_out = y_out.mean(0) # b x layer*hid
            posterior_input = y_out
        else:
            raise NotImplementedError
        return posterior_input

    def encode_table(self, src, fieldmask):
        """
        :param src: b x nfield x 3
        :param fieldmask: b x nfield 0/-inf
        :return: key_emb: b x nfield x 2 x emb
                 masked_key_emb: b x nfield x 2 x emb
                 value_emb: b x nfield x emb
                 h_table_field: b x nfield x hidden
                 h_table: b x hidden
        """
        bsz, nfields, nfeats = src.size()
        emb_size = self.word_emb.embedding_dim
        # src_key, src_value = src[:, :, :2], src[:, :, 2:] # src_key: b x nfield x 2, src_value: b x nfield x 1

        embs = self.word_emb(src.view(-1, nfeats))  # bsz*nfields x nfeats x emb_size

        if self.drop_emb:
            embs = self.drop(embs) # b*nfield x 3 x emb

        key_emb = embs[:, :2, :].view(bsz, nfields, -1, emb_size) # b x nfield x 2 x emb
        value_emb = embs[:, 2:, :].view(bsz, nfields, -1, emb_size).squeeze(2) # b x nfield x emb

        h_table_field = F.tanh(self.table_hidden_out(embs.view(-1, nfeats*emb_size))).view(bsz, nfields, -1) # b x nfield x hidden

        fieldmask[fieldmask == 0] = 1
        fieldmask = fieldmask.unsqueeze(2).expand(bsz, nfields, emb_size).unsqueeze(2).expand(bsz, nfields, 2, emb_size) # b x nfield x 2 x emb
        masked_key_emb = key_emb * fieldmask # b x nfield x 2 x emb

        if self.pool_type == "max":
            masked_key_emb = masked_key_emb.view(bsz, nfields, -1).transpose(1,2) # b x 2*emb x nfield
            masked_key_emb = F.max_pool1d(masked_key_emb, nfields).squeeze(2) # b x 2*emb

            h_table = F.max_pool1d(h_table_field.transpose(1, 2), nfields).squeeze(2) # b x emb

        elif self.pool_type == "mean":
            masked_key_emb = masked_key_emb.mean(1).view(bsz, -1) # b x 2*dim
            h_table = h_table_field.mean(1).view(bsz, -1) # b x emb
        else:
            raise NotImplementedError
        # masked_key_emb = masked_emb[:, :2, :].view(bsz, -1) # b*nfield x 2 x emb -> b x 2*emb

        return key_emb, masked_key_emb, value_emb, h_table_field, h_table

    def decode_pair(self, table_encodes, sentence, template_sent, fieldmask, sentence_mask, valid=False):
        """ training process
        Args:
            table_encodes: outputs of table encoders - key_emb, masked_key_emb, value_emb, h_table
                    key_emb: b x nfield x 2 x emb
                    masked_key_emb: b x 2*emb
                    value_emb: b x nfield  x emb
                    h_table_field: b x nfield x tabhid
                    h_table: b x tabhid
            sentence: seq x b
            template_sent: seq x b (mask <ent> for values and pad to the same length)
            fieldmask: b x nfield
            sentence_mask: seq x b
        """
        key_emb, masked_key_emb, value_emb, h_table_field, h_table = table_encodes

        bsz, nfield, _ = value_emb.size()
        seqlen = sentence.size(0)

        sent_emb = self.word_emb(sentence)  # seq x b x emb
        if self.drop_emb:
            sent_emb = self.drop(sent_emb)  # seq x b x emb
        emb_size = sent_emb.size(2)

        # posterior q(z|x)
        h_y0 = torch.zeros(self.layers*2, bsz, self.hid_size).contiguous() # default bi-rnn, so, 2 layers
        if self.use_cuda:
            h_y0 = h_y0.cuda()

        y_out, h_yt = self.rnn_encode(sent_emb, h_y0)  # y_out: seq x b x layer*hid, h_yt: layer x b x hid

        posterior_input = self._get_posterior_input(y_out, h_yt) # b x layer*hid
        posterior_out_z = self.z_posterior(posterior_input)  # b x latent_z*2
        mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)  # both has size b x latent_z
        # sample z from the posterior
        z_sample = sample_from_gaussian(mu_post_z, logvar_post_z)  # b x latent_z

        # prior of z: p(z) = N(0,I)
        mu_prior_z = self.zeros.expand(z_sample.size())
        logvar_prior_z = self.zeros.expand(z_sample.size())
        # posterior of c q(c|x)
        posterior_out_c = self.c_posterior(posterior_input)
        mu_post_c, logvar_post_c = torch.chunk(posterior_out_c, 2, 1)
        c_sample = sample_from_gaussian(mu_post_c, logvar_post_c)

        # prior of c p(c) = N(0,I)
        mu_prior_c = self.zeros.expand(c_sample.size())
        logvar_prior_c = self.zeros.expand(c_sample.size())
        # generate p(x|z,h_tab) for paired data
        ar_embs = torch.cat([self.word_emb.weight[2].view(1, 1, emb_size).expand(1, bsz, emb_size),
                             sent_emb[:-1, :, :]], 0)  # seqlen x bsz x emb_size

        ar_embs = torch.cat([ar_embs, z_sample.expand(seqlen, bsz, -1), h_table.expand(seqlen, bsz, -1)], dim=-1)  # seq x b x (emb_size + latent_z + tabhid)
        states, (h, c) = self.word_rnn(ar_embs)  # (h0, c0) states: seq x b x hid

        if self.use_dec_attention:
            assert fieldmask is not None
            attn_score_dec, attn_ctx_dec, attn_logits_dec = self.attn_table_hidden.forward(states, h_table_field, h_table_field, fieldmask, return_logits=True)
            # attn_score_dec, attn_logits_dec: b x seq x nfield
            # attn_ctx_dec: b x seq x tabhid
            dec_outs = self.generator_out(torch.cat([states, attn_ctx_dec.transpose(0, 1)], dim=-1)) # seq x b x vocab
        else:
            dec_outs = self.generator_out(states)

        seq_prob = F.softmax(dec_outs, dim=-1)  # seq x b x vocab
        seq_prob = torch.cat([seq_prob, Variable(self.zeros.expand(seq_prob.size(0), seq_prob.size(1), 1))], dim=-1)

        crossEntropy = -torch.log(torch.gather(seq_prob, -1, sentence.view(seqlen, bsz, 1)) + 1e-15)  # seqlen x b x 1
        # print(crossEntropy.squeeze(2).mean())
        # sentence_mask = torch.ByteTensor(sentence.cpu() != self.pad_idx).transpose(0, 1)
        # if self.use_cuda:
        #     sentence_mask = sentence_mask.cuda()
        nll_loss = crossEntropy.masked_select(sentence_mask).mean()

        # KL between prior and posterior KL(q(z|x)||p(z)) and kl(q(c|x)||p(c))
        KL_z = torch.mean(gaussian_kld(mu_post_z, logvar_post_z, mu_prior_z, logvar_prior_z), dim=0)  # dim=b -> mean
        KL_c = torch.mean(gaussian_kld(mu_post_c, logvar_post_c, mu_prior_c, logvar_prior_c), dim=0)  # dim=b -> mean
        if not valid:
            total_loss = nll_loss + self.kl_weight_z * KL_z
        else:
            total_loss = nll_loss + KL_z

        loss_dict = {"pair_nll": nll_loss, "pair_KLz": KL_z, "kl_weight_z": torch.full((1,), self.kl_weight_z)}

        if self.add_preserving_content_loss:
            mse_loss = F.mse_loss(c_sample, h_table)  # L_committment
            if not valid:
                total_loss += self.pc_weight * mse_loss + self.kl_weight_c * KL_c
                loss_dict['pair_mse'] = mse_loss
                loss_dict['pair_KLc'] = KL_c

        if self.add_preserving_template_loss:
            template_emb = self.word_emb(template_sent)
            if self.drop_emb:
                template_emb = self.drop(template_emb)
            ar_embs = torch.cat([self.word_emb.weight[2].view(1, 1, emb_size).expand(1, bsz, emb_size),
                                 template_emb[:-1 ,: ,:]], 0)  # temp_seqlen x bsz x emb_size, init
            temp_seqlen = template_sent.size(0)
            ar_embs = torch.cat([ar_embs, z_sample.expand(temp_seqlen, bsz, -1)], dim=-1)
            temp_states, (h, c) = self.template_rnn(ar_embs)
            temp_outs = self.template_out(temp_states)
            temp_seq_prob  = F.softmax(temp_outs, dim=-1)

            crossEntropy = -torch.log(
                torch.gather(temp_seq_prob, -1, template_sent.view(temp_seqlen, bsz, 1)) + 1e-15)  # seqlen x b x 1
            temp_mask = torch.ByteTensor(template_sent.cpu() != self.pad_idx).transpose(0, 1)
            if self.use_cuda:
                temp_mask = temp_mask.cuda()
            pt_loss = crossEntropy.masked_select(temp_mask).mean()

            if not valid:
                total_loss += self.pt_weight * pt_loss
                loss_dict["pt_loss"] = pt_loss

        # if self.add_skeleton:
        #     sk_sent_emb = self.word_emb(sent_skelton)  # skseq x b x emb
        #     if self.drop_emb:
        #         sk_sent_emb = self.drop(sk_sent_emb)
        #
        #     ar_embs = torch.cat([self.word_emb.weight[2].view(1, 1, emb_size).expand(1, bsz, emb_size), sk_sent_emb[:-1 ,: ,:]], 0)  # skseqlen x bsz x emb_size, init
        #     sk_seqlen = sent_skelton.size(0)
        #     ar_embs = torch.cat([ar_embs, z_sample.expand(sk_seqlen, bsz, -1)], dim=-1)  # skseq x b x (emb_size + latent_z)
        #     sk_states, (h, c) = self.sk_rnn(ar_embs)  # (h0, c0) states: skseq x b x hid
        #     sk_dec_outs = self.sk_generator_out(sk_states) # skseq x b x vocab
        #     sk_seq_prob = F.softmax(sk_dec_outs, dim=-1) # skseq x b x vocab
        #     # nll loss
        #     crossEntropy = -torch.log(torch.gather(sk_seq_prob, -1, sent_skelton.view(sk_seqlen, bsz, 1)) + 1e-15)  # seqlen x b x 1
        #     sent_sk_mask = torch.ByteTensor(sent_skelton.cpu() != self.pad_idx).transpose(0, 1)
        #     if self.use_cuda:
        #         sent_sk_mask = sent_sk_mask.cuda()
        #     skeleton_nllloss = crossEntropy.masked_select(sent_sk_mask).mean()
        #
        #     if not valid:
        #         total_loss += self.sk_weight * skeleton_nllloss
        #         loss_dict['pair_sk_nll'] = skeleton_nllloss

        if self.add_mi_z:
            logqz = norm_log_liklihood(z_sample, self.zeros.expand(z_sample.size()), self.zeros.expand(z_sample.size())) # dim=b
            logqz_Cx = norm_log_liklihood(z_sample, mu_post_z, logvar_post_z) # dim=b
            mutual_info_z = (logqz_Cx - logqz).mean() # b -> 1x1
            if not valid:
                loss_dict['pair_mi_z'] = mutual_info_z
                total_loss += self.mi_z_weight * mutual_info_z

        if self.add_mi_c:
            logqc = norm_log_liklihood(c_sample, self.zeros.expand(c_sample.size()), self.zeros.expand(c_sample.size()))
            logqc_Cx = norm_log_liklihood(c_sample, mu_post_c, logvar_post_c)
            mutual_info_c = (logqc_Cx - logqc).mean()
            if not valid:
                loss_dict['pair_mi_c'] = mutual_info_c
                total_loss += self.mi_c_weight * mutual_info_c

        loss_dict['pair_loss'] = total_loss
        return loss_dict

    def decode_raw(self, sentence, sentence_mask, valid=False):
        # sentence: seq x b
        seqlen, bsz = sentence.size()
        sent_emb = self.word_emb(sentence)  # seq x b x emb
        if self.drop_emb:
            sent_emb = self.drop(sent_emb)  # seq x b x emb
        emb_size = sent_emb.size(2)

        # posterior q(z|x)
        h_y0 = torch.zeros(self.layers * 2, bsz, self.hid_size).contiguous()  # default bi-rnn, so, 2 layers
        if self.use_cuda:
            h_y0 = h_y0.cuda()

        y_out, h_yt = self.rnn_encode(sent_emb, h_y0)  # y_out: seq x b x layer*hid, h_yt: layer x b x hid

        # posterior of z q(z|x)
        posterior_input = self._get_posterior_input(y_out, h_yt)  # b x layer*hid
        posterior_out_z = self.z_posterior(posterior_input)  # b x latent_z*2
        mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)  # both has size b x latent_z
        # sample z from the posterior
        z_sample = sample_from_gaussian(mu_post_z, logvar_post_z)  # b x latent_z

        # prior of z: p(z) = N(0,I)
        mu_prior_z = self.zeros.expand(z_sample.size())
        logvar_prior_z = self.zeros.expand(z_sample.size())
        # posterior of c q(c|x)
        posterior_out_c = self.c_posterior(posterior_input)
        mu_post_c, logvar_post_c = torch.chunk(posterior_out_c, 2, 1)
        c_sample = sample_from_gaussian(mu_post_c, logvar_post_c)

        # prior of c p(c) = N(0,I)
        mu_prior_c = self.zeros.expand(c_sample.size())
        logvar_prior_c = self.zeros.expand(c_sample.size())
        # generate p(x|z,h_tab) for paired data
        # wlps_k_collect = []
        ar_embs = torch.cat([self.word_emb.weight[2].view(1, 1, emb_size).expand(1, bsz, emb_size),
                             sent_emb[:-1, :, :]], 0)  # seqlen x bsz x emb_size
        ar_embs = torch.cat([ar_embs, z_sample.expand(seqlen, bsz, -1), c_sample.expand(seqlen, bsz, -1)],
                            dim=-1)  # seq x b x (emb_size + latent_z + tabhid)
        states, (h, c) = self.word_rnn(ar_embs)  # (h0, c0) states: seq x b x hid

        if self.use_dec_attention:
            raw_attn = torch.zeros(seqlen, bsz, self.c_latent_size).contiguous()
            if self.use_cuda:
                raw_attn = raw_attn.cuda()
            dec_outs = self.generator_out(torch.cat([states, raw_attn], dim=-1)) # seq x b x vocab
        else:
            dec_outs = self.generator_out(states)

        seq_prob = F.softmax(dec_outs)  # seq x b x vocab

        # nll loss
        crossEntropy = -torch.log(torch.gather(seq_prob, -1, sentence.view(seqlen, bsz, 1)) + 1e-15)  # seqlen x b x 1
        nll_loss = crossEntropy.masked_select(sentence_mask).mean()

        # KL between prior and posterior KL(q(z|x)||p(z)) and kl(q(c|x)||p(c))
        KL_z = torch.mean(gaussian_kld(mu_post_z, logvar_post_z, mu_prior_z, logvar_prior_z), dim=0)  # dim=b -> mean
        KL_c = torch.mean(gaussian_kld(mu_post_c, logvar_post_c, mu_prior_c, logvar_prior_c), dim=0)  # dim=b -> mean

        if not valid: # train
            total_loss = nll_loss + self.kl_weight_z * KL_z + self.kl_weight_c * KL_c
        else:
            total_loss = nll_loss + KL_c + KL_z

        loss_dict = {"raw_nll": nll_loss, "raw_KLz": KL_z, "raw_KLc": KL_c,
                     "kl_weight_z": torch.full((1,), self.kl_weight_z), "kl_weight_c": torch.full((1,), self.kl_weight_c)}

        # if self.add_genbow:
        #     p_genbow_fromz = F.softmax(self.gen_bow_z_proj(z_sample), dim=1) # b x gen_voc
        #     p_genbow_fromc = F.softmax(self.gen_bow_c_proj(c_sample), dim=1) # b x gen_voc
        #     sent_genbow = sent_genbow.float()
        #     sent_genbow_p_targ = torch.div((sent_genbow+1).transpose(0, 1), (sent_genbow+1).sum(1)).transpose(0, 1) # normalize b x gen_voc
        #     # we can predict what's in skelton if noly know z
        #     CE_genbow_z = - torch.mean(torch.sum(sent_genbow_p_targ * torch.log(p_genbow_fromz + 1e-15), dim=1), dim=0) # dim=b -> mean
        #     # we cannot predict whats in skelton of only know c
        #     negH_genbow_c = -torch.distributions.categorical.Categorical(probs=p_genbow_fromc).entropy()
        #     negH_genbow_c = torch.mean(negH_genbow_c) # dim=b ->mean
        #     if not valid:
        #         total_loss += self.genbow_weight_z * CE_genbow_z + self.genbow_weight_c * negH_genbow_c
        #         loss_dict['raw_CE_genbow_z'] = CE_genbow_z
        #         loss_dict['raw_H_genbow_c'] = - negH_genbow_c

        # if self.add_vbow:
        #     p_vbow_fromz = F.softmax(self.value_bow_z_proj(z_sample), dim=1)
        #     p_vbow_fromc = F.softmax(self.value_bow_c_proj(c_sample), dim=1)
        #     sent_vbow = sent_vbow.float()
        #     sent_vbow_p_targ = torch.div((sent_vbow+1).transpose(0, 1), (sent_vbow+1).sum(1)).transpose(0, 1)
        #     negH_vbow_z = - torch.distributions.categorical.Categorical(probs=p_vbow_fromz).entropy()
        #     negH_vbow_z = torch.mean(negH_vbow_z, dim=0)
        #     CE_vbow_c = torch.sum(sent_vbow_p_targ * torch.log(p_vbow_fromc + 1e-15), dim=1)
        #     CE_vbow_c = - torch.mean(CE_vbow_c, dim=0)
        #     if not valid:
        #         total_loss += self.vbow_weight_z * negH_vbow_z + self.vbow_weight_c * CE_vbow_c
        #         loss_dict['raw_H_vbow_z'] = - negH_vbow_z
        #         loss_dict['raw_CE_vbow_c'] = CE_vbow_c

        if self.add_mi_z:
            logqz = norm_log_liklihood(z_sample, self.zeros.expand(z_sample.size()), self.zeros.expand(z_sample.size()))  # dim=b
            logqz_Cx = norm_log_liklihood(z_sample, mu_post_z, logvar_post_z)  # dim=b
            mutual_info_z = (logqz_Cx - logqz).mean()  # b -> 1x1
            if not valid:
                loss_dict['pair_mi_z'] = mutual_info_z
                total_loss += mutual_info_z

        if self.add_mi_c:
            logqc = norm_log_liklihood(c_sample, self.zeros.expand(c_sample.size()), self.zeros.expand(c_sample.size()))
            logqc_Cx = norm_log_liklihood(c_sample, mu_post_c, logvar_post_c)
            mutual_info_c = (logqc_Cx - logqc).mean()
            if not valid:
                loss_dict['pair_mi_c'] = mutual_info_c
                total_loss += mutual_info_c

        loss_dict['raw_loss'] = total_loss
        return loss_dict

    def forward(self, pair_src, pair_mask, pair_sentence, pair_sent_skeleton, pair_sent_mask,
                raw_sentence, raw_sent_mask, valid=False):
        # for paired data
        paired_table_enc = self.encode_table(pair_src, pair_mask)
        paired_loss_dict = self.decode_pair(paired_table_enc, pair_sentence, pair_sent_skeleton,
                                            pair_mask, pair_sent_mask, valid=valid)
        # for raw data
        raw_loss_dict = self.decode_raw(raw_sentence, raw_sent_mask, valid=valid)

        all_loss_dict = {}
        for k, v in paired_loss_dict.items():
            all_loss_dict[k] = v
        for k,v in raw_loss_dict.items():
            all_loss_dict[k] = v

        # total loss dict
        if not valid: # train
            total_loss = paired_loss_dict['pair_loss'] + self.rawloss_weight * raw_loss_dict['raw_loss']
        else: # test loss on valid
            total_loss = paired_loss_dict['pair_loss'] + raw_loss_dict['raw_loss']

        return total_loss, all_loss_dict


    def predict(self, paired_src, paired_mask, beam_size=None):
        bsz, _, _ = paired_src.size()

        real_beam_flag = False # default is false
        if beam_size is not None:
            if beam_size == 1:
                self.feedback_x = feedback.GreedyFeedBack(self.word_emb, self.unk_idx)
            else:
                real_beam_flag = True
                self.feedback_x = feedback.BeamFeedBack(self.word_emb, beam_size, self.unk_idx)

        key_emb, masked_key_emb, value_emb, h_table_field, h_table = self.encode_table(paired_src, paired_mask)

        embsz = value_emb.size(-1)

        mu_prior, logvar_prior = torch.zeros(bsz, self.z_latent_size).contiguous(), torch.zeros(bsz, self.z_latent_size).contiguous() # b x latent
        if self.use_cuda:
            mu_prior, logvar_prior = mu_prior.cuda(), logvar_prior.cuda()

        z_sample = sample_from_gaussian(mu_prior, logvar_prior, seed=None)  # b x latent

        # mu_prior_c = self.zeros.expand(bsz, self.c_latent_size)
        # logvar_prior_c = self.zeros.expand(bsz, self.c_latent_size)

        h = self.zeros.expand(1, bsz, self.hid_size).contiguous()
        c = self.zeros.expand(1, bsz, self.hid_size).contiguous()  # to push x makes the use of information from c
        ar_embs = self.word_emb.weight[2].view(1, 1, embsz).expand(1, bsz, embsz)  # 1 x b x emb

        if beam_size is not None:
            past_p = self.zeros.expand(bsz * beam_size, 1) # init past_p for beam search
        else:
            past_p = self.zeros.expand(bsz * self.beamsz, 1)

        for t in range(self.max_seqlen):
            ar_embs = torch.cat([ar_embs, z_sample.unsqueeze(0), h_table.unsqueeze(0)], dim=-1)  # 1 x b x (emb+latent+tabhid)
            ar_state, (h, c) = self.word_rnn(ar_embs, (h, c))

            if self.use_dec_attention:
                attn_score_dec, attn_ctx_dec, attn_logits_dec = self.attn_table_hidden.forward(ar_state, h_table_field,
                                                                                               h_table_field, paired_mask,
                                                                                               return_logits=True)
                # attn_score_dec, attn_logits_dec: b x seq x nfield
                # attn_ctx_dec: b x seq x tabhid
                dec_outs = self.generator_out(torch.cat([ar_state, attn_ctx_dec.transpose(0, 1)], dim=-1))  # seq x b x vocab
            else:
                dec_outs = self.generator_out(ar_state) # seq x b x vocab

            if not real_beam_flag:
                next_inp = self.feedback_x(dec_outs)[0][0].item()
            else:
                word_prob = F.softmax(dec_outs, dim=-1)
                word_prob[:, :, self.unk_idx].fill_(0)  # disallow generating unk word

                cur_p = self.feedback_x.repeat(word_prob).squeeze(0)
                past_p, symbol = self.feedback_x(past_p, cur_p, bsz, t)
                next_inp = symbol[0][0].item()

            next_inp = torch.tensor(next_inp).cuda() if self.use_cuda else torch.tensor(next_inp)
            ar_embs = self.word_emb(next_inp.unsqueeze(0).unsqueeze(1)).expand(1, bsz, embsz)

        if not real_beam_flag:
            sentences_ids = self.feedback_x.collect()
        else:
            sentences_ids = self.feedback_x.collect(past_p, bsz)
        return sentences_ids


    # def predict_from_temp(self, paired_src, paired_mask, temp_sentence):
    #     # sentence: dim = seq
    #     w2i = self.corpus.dictionary.word2idx
    #     bsz, _, _ = paired_src.size()
    #     key_emb, masked_key_emb, value_emb, h_table_field, h_table = self.encode(paired_src, paired_mask)
    #
    #     temp_sentence = temp_sentence.unsqueeze(1) # seq x 1
    #     sent_emb = self.word_emb(temp_sentence)  # seq x 1 x emb
    #     if self.drop_emb:
    #         sent_emb = self.drop(sent_emb)  # seq x 1 x emb
    #
    #     # posterior q(z|x,eK*c)
    #     h_y0 = torch.zeros(self.layers * 2, bsz, self.hid_size).contiguous()  # default bi-rnn, so, 2 layers
    #     if self.use_cuda:
    #         h_y0 = h_y0.cuda()
    #
    #     y_out, h_yt = self.rnn_encode(sent_emb, h_y0)  # y_out: seq x b x layer*hid, h_yt: layer x 1 x hid
    #
    #     # posterior of z q(z|x)
    #     posterior_input = self._get_posterior_input(y_out, h_yt)  # b x layer*hid
    #     posterior_out_z = self.z_posterior(posterior_input)  # b x latent_z*2
    #     mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)  # both has size b x latent_z
    #     # sample z from the posterior
    #     z_sample = sample_from_gaussian(mu_post_z, logvar_post_z)  # b x latent_z
    #
    #     mu_prior_c = self.zeros.expand(bsz, self.c_latent_size)
    #     logvar_prior_c = self.zeros.expand(bsz, self.c_latent_size)
    #     c_sample = sample_from_gaussian(mu_prior_c, logvar_prior_c) # b x latent_c
    #
    #     embsz = sent_emb.size(-1)
    #
    #     h = self.zeros.expand(1, bsz, self.hid_size).contiguous()
    #     c = self.zeros.expand(1, bsz, self.hid_size).contiguous()  # to push x makes the use of information from c
    #     ar_embs = self.word_emb.weight[2].view(1, 1, embsz).expand(1, bsz, embsz)  # 1 x b x emb
    #
    #     for t in range(self.max_seqlen):
    #         # print(ar_embs.size(), z_sample.size())
    #         ar_embs = torch.cat([ar_embs, z_sample.unsqueeze(0), h_table.unsqueeze(0)], dim=-1)  # 1 x b x (emb+latent+tabhid)
    #
    #         # ar_embs = torch.cat([ar_embs, z_sample.unsqueeze(0), c_sample.unsqueeze(0)], dim=-1)  # 1 x b x (emb+latent+tabhid)
    #         ar_state, (h, c) = self.word_rnn(ar_embs, (h, c))
    #
    #         if self.use_dec_attention:
    #             attn_score_dec, attn_ctx_dec, attn_logits_dec = self.attn_table_hidden.forward(ar_state, h_table_field,
    #                                                                                            h_table_field,
    #                                                                                            paired_mask,
    #                                                                                            return_logits=True)
    #             # attn_score_dec, attn_logits_dec: b x seq x nfield
    #             # attn_ctx_dec: b x seq x tabhid
    #             dec_outs = self.generator_out(
    #                 torch.cat([ar_state, attn_ctx_dec.transpose(0, 1)], dim=-1))  # seq x b x vocab
    #         else:
    #             dec_outs = self.generator_out(ar_state)  # seq x b x vocab
    #
    #         next_inp = self.feedback_x(dec_outs)[0][0].item()
    #         # word_prob = F.softmax(dec_outs, dim=-1)
    #         # word_prob[:, :, self.unk_idx].fill_(0)  # disallow generating unk word
    #         # next_inp = self.feedback_x(word_prob)[0][0].item()  # 1x1
    #         next_inp = torch.tensor(next_inp).cuda() if self.use_cuda else torch.tensor(next_inp)
    #         ar_embs = self.word_emb(next_inp.unsqueeze(0).unsqueeze(1)).expand(1, bsz, embsz)
    #         # ar_embs = torch.mean(self.word_emb(next_inp), dim=0).unsqueeze(0).unsqueeze(1).expand(1, bsz, embsz)
    #     sentences_ids = self.feedback_x.collect()
    #     return sentences_ids

    # def inference(self, sentence, return_skeleton=False):
    #     # sentence: seq x 1
    #     # sentence = sentence.unsqueeze(1)  # seq x 1
    #     template_feedback = feedback.SampleFeedBack(self.word_emb, self.unk_idx)
    #     sent_emb = self.word_emb(sentence)  # seq x 1 x emb
    #     if self.drop_emb:
    #         sent_emb = self.drop(sent_emb)  # seq x 1 x emb
    #     seqlen, bsz, emb_size = sent_emb.size() # bsz=1
    #     # posterior q(z|x,eK*c)
    #     h_y0 = torch.zeros(self.layers * 2, bsz, self.hid_size).contiguous()  # default bi-rnn, so, 2 layers
    #     if self.use_cuda:
    #         h_y0 = h_y0.cuda()
    #
    #     y_out, h_yt = self.rnn_encode(sent_emb, h_y0)  # y_out: seq x 1 x layer*hid, h_yt: layer x 1 x hid
    #
    #     # posterior of z q(z|x)
    #     posterior_input = self._get_posterior_input(y_out, h_yt)  # 1 x layer*hid
    #     posterior_out_z = self.z_posterior(posterior_input)  # 1 x latent_z*2
    #     mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)  # both has size 1 x latent_z
    #     # sample z from the posterior
    #     z_sample = sample_from_gaussian(mu_post_z, logvar_post_z)  # 1 x latent_z
    #     z_outs = (posterior_out_z, z_sample)
    #
    #     posterior_out_c = self.c_posterior(posterior_input) # 1 x latent_c*2
    #     mu_post_c, logvar_post_c = torch.chunk(posterior_out_c, 2, 1)
    #     c_sample = sample_from_gaussian(mu_post_c, logvar_post_c) # 1 x latent_c
    #     c_outs = (posterior_out_c, c_sample)
    #
    #     if return_skeleton and self.add_skeleton:
    #         h = self.zeros.expand(1, bsz, self.hid_size).contiguous()
    #         c = self.zeros.expand(1, bsz, self.hid_size).contiguous()  # to push x makes the use of information from c
    #         ar_embs = self.word_emb.weight[2].view(1, 1, self.emb_size).expand(1, bsz, self.emb_size)  # 1 x 1 x emb
    #         for t in range(self.max_seqlen):
    #             # print(ar_embs.size(), z_sample.size())
    #             ar_embs = torch.cat([ar_embs, z_sample.unsqueeze(0)],dim=-1)  # 1 x 1 x (emb+latent_z)
    #             ar_state, (h, c) = self.sk_rnn(ar_embs, (h, c))
    #             dec_outs = self.sk_generator_out(ar_state)  # 1 x 1 x vocab
    #
    #             next_inp = template_feedback(dec_outs)[0][0].item()
    #
    #             # sk_word_prob = F.softmax(dec_outs, dim=-1)
    #             # sk_word_prob[:, :, self.unk_idx].fill_(0)  # disallow generating unk word
    #             # next_inp = self.feedback_x(sk_word_prob)[0][0].item()  # 1x1 scalar
    #             next_inp = torch.tensor(next_inp).cuda() if self.use_cuda else torch.tensor(next_inp)
    #             ar_embs = self.word_emb(next_inp.unsqueeze(0).unsqueeze(1)).expand(1, bsz, self.emb_size)
    #         skeleton_ids = template_feedback.collect()
    #
    #         return z_outs, c_outs, skeleton_ids
    #     else:
    #         return z_outs, c_outs
