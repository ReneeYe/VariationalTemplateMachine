import sys
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from data_utils import label_wiki, label_spnlg
from models.variational_template_machine import VariationalTemplateMachine
import random

def set_optimizer(net):
    if args.optim == "adagrad":
        optalg = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        for group in optalg.param_groups:
            for p in group['params']:
                optalg.state[p]['sum'].fill_(0.1)
    elif args.optim == "rmsprop":
        optalg = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    elif args.optim == "adam":
        optalg = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    else:
        optalg = None
    return optalg

def cuda2cpu():
    if args.cuda:
        shmocals = locals()
        for shk in list(shmocals):
            shv = shmocals[shk]
            if hasattr(shv, "is_cuda") and shv.is_cuda:
                shv = shv.cpu()

def make_skeleton(sentence, sent_len, vocab, gen_dict):
    # sentence: seq x b LongTensor
    # sent_len: dim = b
    totseqlen, bsz = sentence.size()
    skeleton_sents = []
    for b in range(bsz):
        sk = []
        # sk = [vocab.word2idx["<ent>"] for i in range(sent_len[b])]
        for t in range(sent_len[b]):
            if sentence[t, b].item() in gen_dict.idx2word:
                sk.append(sentence[t, b].item())
            else:
                if len(sk) == 0: # first item
                    sk.append(vocab.word2idx['<ent>'])
                elif sk[-1] == vocab.word2idx['<ent>']:
                    continue
                else:
                    sk.append(vocab.word2idx['<ent>'])
        if len(sk) < totseqlen:
            sk.extend([vocab.word2idx['<pad>']]*(totseqlen-len(sk)))

        skeleton_sents.append(sk)
    return torch.LongTensor(skeleton_sents).transpose(0, 1).contiguous() # seq x b

def make_masks(src, pad_idx, max_pool=False):
    """
    src - bsz x nfields x nfeats(3)
    """
    neginf = -1e38
    bsz, nfields, nfeats = src.size()
    fieldmask = (src.eq(pad_idx).sum(2) == nfeats) # binary bsz x nfields tensor
    avgmask = (1 - fieldmask).float() # 1s where not padding
    if not max_pool:
        avgmask.div_(avgmask.sum(1, True).expand(bsz, nfields))
    fieldmask = fieldmask.float() * neginf # 0 where not all pad and -1e38 elsewhere
    return fieldmask, avgmask

def make_sent_msk(sentence, pad_idx):
    return torch.ByteTensor(sentence != pad_idx).transpose(0, 1)


parser = argparse.ArgumentParser(description='')
# basic data setups
parser.add_argument('-data', type=str, default='', help='path to data dir')
parser.add_argument('-bsz', type=int, default=16, help='batch size')
parser.add_argument('-seed', type=int, default=1111, help='set random seed, '
                                                          'when training, it is to shuffle training batch, '
                                                          'when testing, it is to define the latent samples')
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-log_interval', type=int, default=200, help='minibatches to wait before logging training status')
parser.add_argument('-max_vocab_cnt', type=int, default=50000)
parser.add_argument('-max_seqlen', type=int, default=70, help='')

# epochs
parser.add_argument('-epochs', type=int, default=40, help='epochs that train together')
parser.add_argument('-paired_epochs', type=int, default=10, help='epochs that train paired data')
parser.add_argument('-raw_epochs', type=int, default=10, help='epochs that train raw data')
parser.add_argument('-warm_up_epoch', type=int, default=0)

# model saves
parser.add_argument('-load', type=str, default='', help='path to saved model')
parser.add_argument('-save', type=str, default='', help='path to save the model')

# global setups
parser.add_argument('-emb_size', type=int, default=100, help='size of word embeddings')
parser.add_argument('-dropout', type=float, default=0.3, help='dropout')
parser.add_argument('-drop_emb', action='store_true', help='dropout in embedding')
parser.add_argument('-initrange', type=float, default=0.1, help='uniform init interval')
# table encoder setups
parser.add_argument('-table_hid_size', type=int, default=128, help='size of table hidden size')
parser.add_argument('-pool_type', type=str, default="max", help='max/mean pooling')

# sentence embedding setups
parser.add_argument('-hid_size', type=int, default=128, help='size of rnn hidden state')
parser.add_argument('-layers', type=int, default=1, help='num rnn layers')
parser.add_argument('-sent_represent', type=str, default='last_hid', help='last_hid/seq_avg')

# generator setups use attention or not
parser.add_argument('-dec_attention', action='store_true', help='store attention to h when generating')

# latent setups
parser.add_argument('-z_latent_size', type=int, default=200, help="size of latent variable z")
parser.add_argument('-c_latent_size', type=int, default=200, help="size of latent variable c")


# mse loss to train q(c|x)
parser.add_argument('-add_preserving_content_loss', action='store_true', help='add preserving-content loss')
parser.add_argument('-pc_weight', type=float, default=1.0, help='pc loss weight')

# add skeleton loss E_z~q(z|x)logp(x_sk|z)
parser.add_argument('-add_preserving_template_loss', action='store_true', help='add skeleton preserving-template loss')
parser.add_argument('-pt_weight', type=float, default=1.0, help='weight for preserving-template loss')

# annealing tricks for latent variables z and c
parser.add_argument('-anneal_function_z', type=str, default='linear', help='logistic/const/linear')
parser.add_argument('-anneal_k_z', type=float, default=0.1)
parser.add_argument('-anneal_x0_z', type=int, default=6000)
parser.add_argument('-anneal_function_c', type=str, default='linear', help='logistic/const/linear')
parser.add_argument('-anneal_k_c', type=float, default=0.1)
parser.add_argument('-anneal_x0_c', type=int, default=6000)
# additional bow losses (including adverserial loss and multitask loss)
parser.add_argument('-add_mi_z', action='store_true',help='add mi loss for latent z')
parser.add_argument('-add_mi_c', action='store_true', help='add mi loss for latent c')
parser.add_argument('-mi_z_weight', type=float, default=1.0, help='mi weight for z')
parser.add_argument('-mi_c_weight', type=float, default=1.0, help='mi weight for c')

# loss for unlabeled raw data
parser.add_argument('-rawloss_weight', type=float, default=1.0, help='weight for raw data training')

# learning tricks: lr, optimizer
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-lr_decay', type=float, default=0.5, help='learning rate decay')
parser.add_argument('-optim', type=str, default="adam", help='optimization algorithm')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')

# decode method
parser.add_argument('-decode_method', type=str, default='beam_search', help="beam_seach / temp_sample / topk_sample / nucleus_sample")
parser.add_argument('-beamsz', type=int, default=1, help='beam size')
parser.add_argument('-sample_temperature', type=float, default=1.0, help='set sample_temperature for decode_method=temp_sample')
parser.add_argument('-topk', type=int, default=5, help='for topk_sample, if topk=1, it is greedy')
parser.add_argument('-topp', type=float, default=1.0, help='for nucleus(top-p) sampleing, if topp=1, then its fwd_sample')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    sys.stdout.flush()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with -cuda")
            sys.stdout.flush()
        else:
            torch.cuda.manual_seed_all(args.seed)
    else:
        if args.cuda:
            print("No CUDA device.")
            args.cuda = False

    if 'wiki' in args.data.lower():
        corpus = label_wiki.Corpus(args.data, args.bsz, max_count=args.max_vocab_cnt,
                                   add_bos=False, add_eos=False)
    elif 'spnlg' in args.data.lower():
        corpus = label_spnlg.Corpus(args.data, args.bsz, max_count=args.max_vocab_cnt,
                                    add_bos=False, add_eos=True)
    else:
        raise NotImplementedError

    print("data loaded!")
    print("total vocabulary size:", len(corpus.dictionary))
    args.pad_idx = corpus.dictionary.word2idx['<pad>']

    if len(args.load) > 0:
        print("load model ...")
        saved_stuff = torch.load(args.load)
        saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
        for k, v in args.__dict__.items():
            if k not in saved_args.__dict__:
                saved_args.__dict__[k] = v
            if k in ["decode_method", "beamsz", "sample_temperature", "topk", "topp"]:
                saved_args.__dict__[k] = v
        net = VariationalTemplateMachine(corpus, saved_args)
        net.load_state_dict(saved_state, strict=False)
        del saved_args, saved_state, saved_stuff
    else:
        net = VariationalTemplateMachine(corpus, args)
    if args.cuda:
        net = net.cuda()

    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Num of parameters in vae model: {:.2f} M. ".format(trainable_num/1000/1000))
    optalg = set_optimizer(net)


    def train_pair(epoch):
        net.train()
        global tot_steps
        loss_record = defaultdict(float)
        nsents = 0
        trainperm = torch.randperm(len(corpus.paired_train))

        for batch_idx in range(len(corpus.paired_train)):
            net.zero_grad()
            sentence, paired_skeleton_sent, paired_sentlen, paired_src_feat = corpus.paired_train[trainperm[batch_idx]]
            paired_mask, _ = make_masks(paired_src_feat, args.pad_idx)
            sentence_mask = make_sent_msk(sentence, args.pad_idx)

            # if not args.add_genbow and not args.add_vbow:
            #     paired_genbow, paired_vbow = None, None
            # else:
            #     paired_genbow, paired_vbow = make_bow(sentence, paired_sentlen, corpus.value_dict, corpus.gen_vocab)
            #     if args.cuda:
            #         paired_genbow, paired_vbow = paired_genbow.cuda(), paired_vbow.cuda()
            if args.cuda:
                paired_src_feat, paired_mask = paired_src_feat.cuda(), paired_mask.cuda()
                sentence, paired_skeleton_sent, sentence_mask = sentence.cuda(), paired_skeleton_sent.cuda(), sentence_mask.cuda()

            paired_table_enc = net.encode_table(Variable(paired_src_feat), Variable(paired_mask))

            tot_steps += 1
            if hasattr(net, 'set_kl_weight'):
                net.set_kl_weight(tot_steps)

            paired_loss_dict = net.decode_pair(paired_table_enc, Variable(sentence),
                                               Variable(paired_skeleton_sent), Variable(paired_mask),
                                               Variable(sentence_mask), valid=False)
            loss = paired_loss_dict['pair_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            if optalg is not None:
                optalg.step()
            else:
                for p in net.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.lr, p.grad.data)

            for l, v in paired_loss_dict.items():
                loss_record[l] += v.item()
            nsents += 1

            if (batch_idx+1) % args.log_interval == 0:
                log_str = "batch %d/%d " % (batch_idx+1, len(corpus.paired_train))
                for k, v in paired_loss_dict.items():
                    log_str += ("| train %s %g ") % (k, v.item())
                    # writer.add_scalar('Train/{}'.format(k), v.item(), tot_steps)
                print(log_str)
                sys.stdout.flush()


            del sentence, paired_src_feat, paired_mask, paired_sentlen, paired_loss_dict

        log_str = "paired epoch %d " % (epoch)
        for k, v in loss_record.items():
            log_str += ("| train %s %g ") % (k, v / nsents)
        print(log_str)

    def train_raw(epoch):
        net.train()
        global tot_steps
        tot_loss = 0.0
        loss_record = defaultdict(float)
        nsents = 0
        trainperm = torch.randperm(len(corpus.raw_train))

        for batch_idx in range(len(corpus.raw_train)):
            net.zero_grad()
            raw_sentence, raw_sentlen = corpus.raw_train[trainperm[batch_idx]]
            sentence_mask = make_sent_msk(raw_sentence, args.pad_idx)
            # if not args.add_genbow and not args.add_vbow:
            #     raw_genbow, raw_vbow = None, None
            # else:
            #     raw_genbow, raw_vbow = make_bow(raw_sentence, raw_sentlen, corpus.value_dict, corpus.gen_vocab)
            #     if args.cuda:
            #         raw_genbow, raw_vbow = raw_genbow.cuda(), raw_vbow.cuda()

            if args.cuda:
                raw_sentence, raw_sentlen = raw_sentence.cuda(), raw_sentlen.cuda()
                sentence_mask = sentence_mask.cuda()

            raw_loss_dict = net.decode_raw(Variable(raw_sentence), Variable(sentence_mask), valid=False)
            # raw_loss_dict = net.decode_raw(Variable(raw_sentence), Variable(raw_genbow), Variable(raw_vbow), Variable(sentence_mask), valid=False)

            loss = raw_loss_dict['raw_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            if optalg is not None:
                optalg.step()
            else:
                for p in net.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.lr, p.grad.data)
            for l, v in raw_loss_dict.items():
                loss_record[l] += v.item()
            nsents += 1

            if (batch_idx+1) % args.log_interval == 0:
                log_str = "batch %d/%d " % (batch_idx+1, len(corpus.raw_train))
                for k, v in raw_loss_dict.items():
                    log_str += ("| train %s %g ") % (k, v.item())
                    # writer.add_scalar('Train/{}'.format(k), v.item(), tot_steps)
                print(log_str)

            del raw_sentence, raw_sentlen, raw_loss_dict

        log_str = "raw epoch %d " % (epoch)
        for k, v in loss_record.items():
            log_str += ("| train %s %g ") % (k, v / nsents)
        print(log_str)
        sys.stdout.flush()

    def train_together(epoch):
        net.train()
        global tot_steps
        nsents = 0
        loss_record = defaultdict(float)
        train_size = min(len(corpus.paired_train), len(corpus.raw_train))

        paired_perm = np.random.choice(len(corpus.paired_train), size=train_size, replace=False)
        raw_perm = np.random.choice(len(corpus.raw_train), size=train_size, replace=False)

        for batch_idx in range(train_size):
            net.zero_grad()
            # load pair data
            pair_sentence, paired_skeleton_sent, paired_sentlen, paired_src_feat = corpus.paired_train[paired_perm[batch_idx]]
            paired_mask, _ = make_masks(paired_src_feat, args.pad_idx)
            paired_sentence_mask = make_sent_msk(pair_sentence, args.pad_idx)
            # if not args.add_genbow and not args.add_vbow:
            #     paired_genbow, paired_vbow = None, None
            # else:
            #     paired_genbow, paired_vbow = make_bow(pair_sentence, paired_sentlen, corpus.value_dict, corpus.gen_vocab)
            #     if args.cuda:
            #         paired_genbow, paired_vbow = paired_genbow.cuda(), paired_vbow.cuda()

            # load raw data
            raw_sentence, raw_sentlen = corpus.raw_train[raw_perm[batch_idx]]
            raw_sentence_mask = make_sent_msk(raw_sentence, args.pad_idx)
            # if not args.add_genbow and not args.add_vbow:
            #     raw_genbow, raw_vbow = None, None
            # else:
            #     raw_genbow, raw_vbow = make_bow(raw_sentence, raw_sentlen, corpus.value_dict, corpus.gen_vocab)
            #     if args.cuda:
            #         raw_genbow, raw_vbow = raw_genbow.cuda(), raw_vbow.cuda()

            if args.cuda:
                paired_src_feat, paired_mask = paired_src_feat.cuda(), paired_mask.cuda()
                pair_sentence, paired_skeleton_sent = pair_sentence.cuda(), paired_skeleton_sent.cuda()
                paired_sentence_mask = paired_sentence_mask.cuda()
                raw_sentence, raw_sentlen = raw_sentence.cuda(), raw_sentlen.cuda()
                raw_sentence_mask = raw_sentence_mask.cuda()

            tot_steps += 1
            if hasattr(net, 'set_kl_weight'):
                net.set_kl_weight(tot_steps)

            total_loss, all_loss_dict = net.forward(Variable(paired_src_feat), Variable(paired_mask), Variable(pair_sentence),
                                                    Variable(paired_skeleton_sent), Variable(paired_sentence_mask),
                                                    Variable(raw_sentence), Variable(raw_sentence_mask), valid=False)

            # total_loss, all_loss_dict = net.forward(Variable(paired_src_feat), Variable(paired_mask), Variable(pair_sentence),
            #                                         Variable(paired_skeleton_sent), Variable(paired_genbow), Variable(paired_vbow), Variable(paired_sentence_mask),
            #                                         Variable(raw_sentence), Variable(raw_genbow), Variable(raw_vbow), Variable(raw_sentence_mask),
            #                                         valid=False)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            if optalg is not None:
                optalg.step()
            else:
                for p in net.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.lr, p.grad.data)

            for l, v in all_loss_dict.items():
                loss_record[l] += v.item()
            nsents += 1

            if (batch_idx+1) % args.log_interval == 0:
                log_str = "batch %d/%d " % (batch_idx+1, train_size)
                for k, v in all_loss_dict.items():
                    log_str += ("| train %s %g ") % (k, v.item())
                    # writer.add_scalar('Train/{}'.format(k), v.item(), tot_steps)
                print(log_str)

            del pair_sentence, paired_src_feat, paired_sentlen, paired_mask, paired_skeleton_sent
            del raw_sentlen, raw_sentence, all_loss_dict

        log_str = "together epoch %d " % (epoch)
        for k, v in loss_record.items():
            log_str += ("| train %s %g ") % (k, v / nsents)
        print(log_str)

    def valid(epoch):
        net.eval()
        pairnsents = 0
        rawnsents = 0
        loss_record = defaultdict(float)
        for i in range(len(corpus.paired_valid)):
            paired_sentence, paired_skeleton_sent, paired_sentlen, paired_src_feat = corpus.paired_valid[i]
            paired_mask, _ = make_masks(paired_src_feat, args.pad_idx)
            sentence_mask = make_sent_msk(paired_sentence, args.pad_idx)
            # if not args.add_genbow and not args.add_vbow:
            #     paired_genbow, paired_vbow = None, None
            # else:
            #     paired_genbow, paired_vbow = make_bow(paired_sentence, paired_sentlen, corpus.value_dict, corpus.gen_vocab)
            #     if args.cuda:
            #         paired_genbow, paired_vbow = paired_genbow.cuda(), paired_vbow.cuda()

            if args.cuda:
                paired_src_feat, paired_mask = paired_src_feat.cuda(), paired_mask.cuda()
                paired_sentence, paired_skeleton_sent = paired_sentence.cuda(), paired_skeleton_sent.cuda()
                sentence_mask = sentence_mask.cuda()

            paired_table_enc = net.encode_table(paired_src_feat, paired_mask)
            paired_loss_dict = net.decode_pair(paired_table_enc, paired_sentence, paired_skeleton_sent, paired_mask,
                                               sentence_mask, valid=True)

            for l, v in paired_loss_dict.items():
                loss_record[l] += v.item()
            pairnsents += 1

            del paired_sentence, paired_src_feat, paired_mask, paired_skeleton_sent, paired_loss_dict

        if corpus.raw_valid is None:
            log_str = "No raw data for valid; epoch %d " % (epoch)
            for k, v in loss_record.items():
                log_str += ("| valid %s %g ") % (k, v / pairnsents)
            print(log_str)
            sys.stdout.flush()
            return loss_record['pair_loss'] / pairnsents

        else:
            for i in range(len(corpus.raw_valid)):
                raw_sentence, raw_sentlen = corpus.raw_valid[i]
                sentence_mask = make_sent_msk(raw_sentence, args.pad_idx)
                # if not args.add_genbow and not args.add_vbow:
                #     raw_genbow, raw_vbow = None, None
                # else:
                #     raw_genbow, raw_vbow = make_bow(raw_sentence, raw_sentlen, corpus.value_dict, corpus.gen_vocab)
                #     if args.cuda:
                #         raw_genbow, raw_vbow = raw_genbow.cuda(), raw_vbow.cuda()
                if args.cuda:
                    raw_sentence, raw_sentlen = raw_sentence.cuda(), raw_sentlen.cuda()
                    sentence_mask = sentence_mask.cuda()

                raw_loss_dict = net.decode_raw(raw_sentence, sentence_mask, valid=True)
                for l, v in raw_loss_dict.items():
                    loss_record[l] += v.item()
                rawnsents += 1

            log_str = "epoch %d " % (epoch)
            for k, v in loss_record.items():
                if "pair" in k:
                    log_str += ("| valid %s %g ") % (k, v / pairnsents)
                    # writer.add_scalar('Valid/{}'.format(k), v / pairnsents, epoch)
                if "raw" in k:
                    log_str += ("| valid %s %g ") % (k, v / rawnsents)
                    # writer.add_scalar('Valid/{}'.format(k), v / rawnsents, epoch)
            print(log_str)
            valid_loss = loss_record["pair_loss"] / pairnsents + loss_record["raw_loss"] / rawnsents
            return valid_loss

    def generate_samples(num=3):
        net.eval()
        valid_candidates = random.sample(range(len(corpus.paired_valid)), num)
        for i in valid_candidates:
            paired_sentence, paired_skeleton_sent, _, paired_src_feat = corpus.paired_valid[i]
            src_str = ""
            for keyid, widx in zip(paired_src_feat[0, :, 0], paired_src_feat[0, :, 2]):
                src_str += corpus.dictionary.idx2word[keyid] + "_" + corpus.dictionary.idx2word[widx] + "|"
            print("Source: ", src_str)

            ref = []
            for t in range(paired_sentence.size(0)):
                word = corpus.dictionary.idx2word[paired_sentence[t, 0].item()]
                if word != '<pad>':
                    ref.append(word)
            print("Reference: {}".format(" ".join(ref)))

            paired_mask, _ = make_masks(paired_src_feat, args.pad_idx)
            if args.cuda:
                paired_src_feat, paired_mask = paired_src_feat.cuda(), paired_mask.cuda()
            sentence_ids = net.predict(paired_src_feat, paired_mask)
            sentence_ids = sentence_ids.data.cpu()
            sent_words = []
            for t, wid in enumerate(sentence_ids[:, 0]):
                word = corpus.dictionary.idx2word[wid]
                sent_words.append(word)
            print("Predict: {}".format(" ".join(str(w) for w in sent_words)))


    prev_valloss, best_valloss = float("inf"), float("inf")
    tot_steps = 0

    total_epoch = 1
    for epoch in range(1, args.paired_epochs + 1):
        train_pair(epoch)
        valloss = valid(epoch)
        generate_samples() # show some generation samples
        if valloss < best_valloss and total_epoch > args.warm_up_epoch:
            print("save best valid loss = {:.4f}".format(valloss))
            best_valloss = valloss
            if len(args.save) > 0:
                print("saving to {}...".format(args.save))
                state = {"opt": args, "state_dict": net.state_dict(), "lr": args.lr,
                         "all_dict": corpus.dictionary, "gen_dict": corpus.gen_vocab,
                         "value_dict": corpus.value_dict}
                torch.save(state, args.save)
        prev_valloss = valloss
        cuda2cpu()
        total_epoch += 1

    for epoch in range(1, args.raw_epochs + 1):
        train_raw(epoch)
        valloss = valid(epoch)
        generate_samples()
        if valloss < best_valloss and total_epoch > args.warm_up_epoch:
            print("save best valid loss = {:.4f}".format(valloss))
            best_valloss = valloss
            if len(args.save) > 0:
                print("saving to {}...".format(args.save))
                state = {"opt": args, "state_dict": net.state_dict(), "lr": args.lr,
                         "all_dict": corpus.dictionary, "gen_dict": corpus.gen_vocab,
                         "value_dict": corpus.value_dict}
                torch.save(state, args.save)
        prev_valloss = valloss
        cuda2cpu()
        total_epoch += 1

    for epoch in range(1, args.epochs + 1):
        train_together(epoch)
        valloss = valid(epoch)
        generate_samples()
        if valloss < best_valloss and total_epoch > args.warm_up_epoch:
            print("save best valid loss = {:.4f}".format(valloss))
            best_valloss = valloss
            if len(args.save) > 0:
                print("saving to {}...".format(args.save))
                state = {"opt": args, "state_dict": net.state_dict(), "lr": args.lr,
                         "all_dict": corpus.dictionary, "gen_dict": corpus.gen_vocab,
                         "value_dict": corpus.value_dict}
                torch.save(state, args.save)
        prev_valloss = valloss
        cuda2cpu()
        total_epoch += 1
