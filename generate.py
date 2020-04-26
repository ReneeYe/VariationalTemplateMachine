import sys
import argparse
import numpy as np
import torch
from data_utils import label_wiki, label_spnlg
from models import variational_template_machine
import os

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

def random_mask(src, pad_idx, prob=0.4, seed=1):
    # src: b x nfield x 3(key, pos, wrd)
    neginf = -1e38
    bsz, nfield, _ = src.size()
    fieldmask = (src.eq(pad_idx).sum(2) == 3) # b x nfield, 0 for has, 1 for pad
    mask_matrix = torch.rand(bsz, nfield, generator=torch.manual_seed(seed))
    mask_matrix = torch.max((mask_matrix < prob), fieldmask) # 1 for pad, and 0 for not pad
    mask_matrix = mask_matrix.float() * neginf # 0 for not pad, -inf for pad
    return mask_matrix


parser = argparse.ArgumentParser(description='')
# basic data setups
parser.add_argument('-data', type=str, default='', help='path to data dir')
parser.add_argument('-bsz', type=int, default=16, help='batch size')
parser.add_argument('-seed', type=int, default=1111, help='set random seed, '
                                                          'when training, it is to shuffle training batch, '
                                                          'when testing, it is to define the latent samples')
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-max_vocab_cnt', type=int, default=50000)
parser.add_argument('-max_seqlen', type=int, default=70, help='')

# model saves
parser.add_argument('-load', type=str, default='', help='path to saved model')

# for generation and test
parser.add_argument('-gen_to_fi', type=str, default=None, help='generate to which file')
parser.add_argument('-various_gen', type=int, default=1, help='define generation how many sentence, and the result is saved in gen_to_fi')
parser.add_argument('-mask_prob', type=float, default=0.0, help='mask item at prob')

# decode method
parser.add_argument('-decode_method', type=str, default='beam_search', help="beam_seach/temp_sample/topk_sample/nucleus_sample")
parser.add_argument('-beamsz', type=int, default=1, help='beam size')
parser.add_argument('-sample_temperature', type=float, default=1.0, help='set sample_temperature for decode_method=temp_sample')
parser.add_argument('-topk', type=int, default=5, help='for topk_sample, if topk=1, it is greedy')
parser.add_argument('-topp', type=float, default=1.0, help='for nucleus(top-p) sampleing, if topp=1, then its fwd_sample')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
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

    # data loader
    if 'wiki' in args.data.lower():
        corpus = label_wiki.Corpus(args.data, args.bsz, max_count=args.max_vocab_cnt,
                                   add_bos=False, add_eos=False)
    elif 'spnlg' in args.data.lower():
        corpus = label_spnlg.Corpus(args.data, args.bsz, max_count=args.max_vocab_cnt,
                                    add_bos=False, add_eos=True)
    else:
        raise NotImplementedError
    print("data loaded!")
    print("Vocabulary size:", len(corpus.dictionary))
    args.pad_idx = corpus.dictionary.word2idx['<pad>']

    # load model
    if len(args.load) > 0:
        print("load model ...")
        saved_stuff = torch.load(args.load)
        saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
        for k, v in args.__dict__.items():
            if k not in saved_args.__dict__:
                saved_args.__dict__[k] = v
            if k in ["decode_method", "beamsz", "sample_temperature", "topk", "topp"]:
                saved_args.__dict__[k] = v
        net = variational_template_machine.VariationalTemplateMachine(corpus, saved_args)
        net.load_state_dict(saved_state, strict=False)
        del saved_args, saved_state, saved_stuff
    else:
        print("WARNING: No model load! Random init.")
        net = variational_template_machine.VariationalTemplateMachine(corpus, args)
    if args.cuda:
        net = net.cuda()

    def generation(test_out, num=3):
        output_fn = open(test_out, 'w')
        # read source table
        table_path = os.path.join(args.data, "src_test.txt")
        paired_src_feat_tst, origin_src_tst, lineno_tst = corpus.get_test_data(table_path)
        for i in range(len(paired_src_feat_tst)):
            paired_src_feat = paired_src_feat_tst[i]
            for j in range(num):
                if j == 0:
                    paired_mask, _ = make_masks(paired_src_feat, args.pad_idx)
                else: # you may set args.mask_prob=0
                    paired_mask = random_mask(paired_src_feat.cpu(), args.pad_idx, prob=args.mask_prob,
                                              seed=np.random.randint(5000))
                if args.cuda:
                    paired_src_feat, paired_mask = paired_src_feat.cuda(), paired_mask.cuda()
                if args.decode_method != "beam_search":
                    sentence_ids = net.predict(paired_src_feat, paired_mask)
                else:
                    sentence_ids = net.predict(paired_src_feat, paired_mask, beam_size=j+1)

                sentence_ids = sentence_ids.data.cpu()
                sent_words = []
                for t, wid in enumerate(sentence_ids[:, 0]):
                    word = corpus.dictionary.idx2word[wid]
                    if word != '<eos>':
                        sent_words.append(word)
                    else:
                        break
                output_fn.write(" ".join(str(w) for w in sent_words) + '\n')
            output_fn.write("\n")
        output_fn.close()

    net.eval()
    generation(args.gen_to_fi, num=args.various_gen)
