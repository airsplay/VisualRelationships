import torch
import json
import os
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
from model import \
    SpeakerEncoder, SpeakerDecoder, MultiCtxEncoder, MultiCtxDecoder, \
    CrossAttEncoder, SpeakerDecoderTran, NewCtxDecoder, NewAttEncoder, \
    LinearAct
from decoders import DynamicDecoder, DynamicDecoderFC, DynamicDecoderMH, DynamicDecoderMHC
from tensorboardX import SummaryWriter
from param import args
import numpy as np
from evaluate import LangEvaluator
import utils
from tqdm import tqdm

class Speaker:
    def __init__(self, dataset):
        # Built up the model
        self.tok = dataset.tok
        self.feature_size = 2048

        if args.model == 'init':
            self.encoder = SpeakerEncoder(self.feature_size).cuda()
            ctx_size = self.feature_size
            self.decoder = SpeakerDecoderTran(self.tok.vocab_size, ctx_size, heads=1).cuda()
        elif args.model == 'heads':
            self.encoder = MultiCtxEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = MultiCtxDecoder(self.tok.vocab_size, ctx_size, heads=2).cuda()
        elif args.model == 'crossatt':
            self.encoder = CrossAttEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = MultiCtxDecoder(self.tok.vocab_size, ctx_size, heads=2).cuda()
        elif args.model == 'newheads':
            self.encoder = MultiCtxEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = NewCtxDecoder(self.tok.vocab_size, ctx_size, heads=2).cuda()
        elif args.model == 'newcross':
            self.encoder = NewAttEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = NewCtxDecoder(self.tok.vocab_size, ctx_size, heads=2).cuda()
        elif args.model == 'dynamic':
            self.encoder = MultiCtxEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = DynamicDecoderFC(self.tok.vocab_size, ctx_size).cuda()
        elif args.model == 'dynamicmh':
            self.encoder = MultiCtxEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = DynamicDecoderMH(self.tok.vocab_size, ctx_size).cuda()
        elif args.model == 'dmc':
            self.encoder = MultiCtxEncoder(self.feature_size).cuda()
            ctx_size = self.encoder.ctx_dim
            self.decoder = DynamicDecoderMHC(self.tok.vocab_size, ctx_size).cuda()

        if args.baseline == 'linear':
            self.critic = LinearAct(args.hid_dim, 1).cuda()

        # Optimizer
        self.optim = args.optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                    lr=args.lr)

        # Logs
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output)     # Tensorboard summary writer

        # Loss
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.pad_id)

    def train(self, train_tuple, eval_tuple, num_epochs, rl=False):
        train_ds, train_tds, train_loader = train_tuple
        best_eval_score = -95
        best_minor_score = -95

        train_evaluator = LangEvaluator(train_ds)
        reward_func = lambda uidXpred: train_evaluator.get_reward(uidXpred, args.metric)

        for epoch in range(num_epochs):
            print()
            iterator = tqdm(enumerate(train_loader), total=len(train_tds)//args.batch_size, unit="batch")
            word_accu = 0.
            for i, (uid, src, trg, inst, leng) in iterator:
                inst = utils.cut_inst_with_leng(inst, leng)
                # utils.show_case(src[0], trg[0], inst[0], self.tok, os.path.expanduser("~/tmp/speaker/train%d" % i))
                src, trg, inst = src.cuda(), trg.cuda(), inst.cuda()
                self.optim.zero_grad()
                if rl:
                    loss, batch_word_accu = self.rl_training(
                        uid, src, trg, inst, leng, reward_func
                    )
                else:
                    loss, batch_word_accu = self.teacher_forcing(src, trg, inst, leng, train=True)
                word_accu += batch_word_accu
                iterator.set_postfix(loss=loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.)
                self.optim.step()
            word_accu /= (i + 1)        # i is the number of batches, if batch_size != 0
            print("Epoch %d, Training Word Accuracy %0.4f" % (epoch, word_accu))

            if epoch % 1 == 0:
                print("Epoch %d" % epoch)

                # cal sentence level metrics
                scores, uid2pred = self.evaluate(eval_tuple)
                main_metric = args.metric
                main_score = scores[main_metric]

                minor_metric = 'word_accu'
                minor_score = scores[minor_metric]

                # Log
                self.writer.add_scalar(main_metric, main_score, epoch)
                log_str = ""
                for key in sorted(scores.keys()):
                    log_str += " %s: %0.4f " % (key, scores[key])

                if main_score > best_eval_score or minor_score > best_minor_score:
                    if main_score > best_eval_score:
                        print("New best result of %s at %0.4f. Save model and dump result" % (main_metric, main_score))
                        self.save("best_eval")
                        best_log_str = log_str
                        with open(os.path.join(self.output, 'best_eval_score.log'), 'a') as f:
                            f.write('Epoch %d\n' % epoch)
                            f.write('BEST is %s\n' % best_log_str)
                        best_eval_score = main_score
                        json.dump(uid2pred, open(os.path.join(self.output, "best_eval_pred.json"), 'w'), indent=4, sort_keys=True)
                    if minor_score > best_minor_score:
                        with open(os.path.join(self.output, 'best_minor_score.log'), 'a') as f:
                            f.write('Epoch %d\n' % epoch)
                            f.write('BEST is %s\n' % best_log_str)
                        best_minor_score = minor_score
                        json.dump(uid2pred, open(os.path.join(self.output, "best_minor_pred.json"), 'w'), indent=4, sort_keys=True)
                if args.pretrain:       # save the snap each epoch in pretrain mode
                    self.save("eval_%d" % epoch)
                json.dump(uid2pred, open(os.path.join(self.output, "eval_pred_%d.json" % epoch), 'w'), indent=4, sort_keys=True)
                print("BEST is %s" % best_log_str)
                print(log_str)


    def rl_training(self, uid, src, trg, inst, leng, reward_func,
                          ml_w=0.05, rl_w=1., e_w=0.005):
        self.encoder.train()
        self.encoder.resnet_extractor.eval()
        self.decoder.train()
        loss = 0.

        # RL_training
        # Sampling a batch
        insts, log_probs, entropies, hiddens = self.infer_batch(
            src, trg, sampling=True, train=True
        )
        mask = torch.from_numpy((insts[:, 1:] != self.tok.pad_id).astype(np.float32)).cuda()
        tokens = mask.sum()     # Dominate by tokens just like ML

        # Get reward
        sents = [self.tok.decode(self.tok.shrink(inst)) for inst in insts]
        uidXsents = list(zip(uid, sents))
        batch_reward = reward_func(uidXsents)
        batch_reward = torch.from_numpy(batch_reward).cuda()

        # Get baseline
        if args.baseline == 'none':
            baseline = 0.
        elif args.baseline == 'self':
            max_insts = self.infer_batch(
                src, trg, sampling=False, train=False
            )
            self.encoder.train()
            self.decoder.train()
            max_sents = [self.tok.decode(self.tok.shrink(inst)) for inst in max_insts]
            uidXsents = list(zip(uid, max_sents))
            max_reward = reward_func(uidXsents)
            baseline = torch.from_numpy(max_reward).unsqueeze(1).cuda()
        elif args.baseline == 'linear':
            baseline = self.critic(hiddens.detach()).squeeze(2)
            loss += 0.5 * (((baseline - batch_reward.unsqueeze(1)) ** 2) * mask).sum() / tokens
        else:
            assert False

        # Calculate loss
        entropy = (- entropies * mask).sum() / tokens
        rl_loss = ((batch_reward.unsqueeze(1) - baseline.detach()) * (- log_probs) * mask).sum() / tokens

        # ML_training
        ml_loss, word_accu = self.teacher_forcing(src, trg, inst, leng, train=True)

        loss += ml_w * ml_loss + rl_w * rl_loss + e_w * entropy
        return loss, word_accu


    def teacher_forcing(self, src, trg, inst, leng, train=True):
        """
        :param src:  src images (b x 224 x 224 x 3)?
        :param trg:  trg images
        :param inst: encoded sentences (b x max_len)
        :param leng: lengths of sentences (b)
        :param train: dropout or not?
        :return:
        """
        if train:
            self.encoder.train()
            self.encoder.resnet_extractor.eval()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Encoder
        ctx = self.encoder(src, trg)

        # Decoder
        batch_size = inst.size(0)
        h_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        logits, h1, c1 = self.decoder(inst, h_t, c_t, ctx, None)

        # softmax cross entropy loss
        loss = self.softmax_loss(
            input  = logits[:, :-1, :].contiguous().view(-1, self.tok.vocab_size),         # -1 for aligning
            target = inst[:, 1:].contiguous().view(-1)              # "1:" to ignore the word <BOS>
        )

        # Word Accuracy
        _, predict_words = logits.max(2)   # B, l
        correct = (predict_words[:, :-1] == inst[:, 1:])
        word_accu = correct.sum().item() / (inst != self.tok.pad_id).sum().item()

        return loss, word_accu

    def infer_batch(self, src, trg, sampling=True, train=False):
        """
        :param src:  src images (b x 224 x 224 x 3)?
        :param trg:  trg images
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        batch_size = src.size(0)

        # Encoder
        ctx = self.encoder(src, trg)

        # Decoder
        word = np.ones(batch_size, np.int64) * self.tok.bos_id    # First word is <BOS>
        words = [word]
        h_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        ended = np.zeros(batch_size, np.bool)
        word = torch.from_numpy(word).view(-1, 1).cuda()
        log_probs = []
        hiddens = []
        entropies = []
        device = torch.device('cpu')
        for i in range(args.max_input):
            # Decode Step
            logits, h_t, c_t = self.decoder(word, h_t, c_t, ctx, None)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                       # logits: (b, vocab_size)
            # if not sampling:
            logits[:, self.tok.unk_id] = -float("inf")                      # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                # print(logits)
                # print(logits.max())
                m = Categorical(probs)
                word = m.sample()
                if train:
                    log_probs.append(m.log_prob(word))
                    hiddens.append(h_t)
                    entropies.append(m.entropy())
            else:
                values, word = logits.max(1)

            # Append the word
            cpu_word = word.to(device).numpy()
            cpu_word[ended] = self.tok.pad_id
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.eos_id)
            if ended.all():
                break
        if train:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(entropies, 1), \
                   torch.stack(hiddens, 1)
        else:
            return np.stack(words, 1)       # [(b), (b), (b), ...] --> [b, l]

    def evaluate(self, eval_tuple, iters=-1):
        dataset, th_dset, dataloader = eval_tuple
        evaluator = LangEvaluator(dataset)

        # Generate sents by neural speaker
        all_insts = []
        all_gts = []
        uids = []
        word_accu = 0.
        for i, (uid, src, trg, inst, leng) in enumerate(dataloader):
            if i == iters:
                break
            # utils.show_case(src[0], trg[0], inst[0], self.tok, os.path.expanduser("~/tmp/speaker/valid%d" % i))
            src, trg = src.cuda(), trg.cuda()

            # get sentence level predictions
            infer_inst = self.infer_batch(src, trg, sampling=False, train=False)
            all_insts.extend(infer_inst)
            all_gts.extend(inst.cpu().numpy())
            uids.extend(uid)

            # Calculate word level accuracy
            inst = inst.cuda()
            _, batch_accu = self.teacher_forcing(src, trg, inst, leng, train=False)
            word_accu += batch_accu
        word_accu /= (i + 1)

        # Show the gt and predict seq
        for _ in range(3):
            import random
            i = random.randint(0, len(all_gts)-1)
            print('GT:   ' + self.tok.decode(self.tok.shrink(all_gts[i])))
            print('Pred: ' + self.tok.decode(self.tok.shrink(all_insts[i])))

        # For computing the MsCOCO scores
        assert len(uids) == len(all_insts) == len(all_gts)
        # print(len(uids))
        # print(len(all_insts))
        uid2pred = {uid: self.tok.decode(self.tok.shrink(pred))
                    for (uid, pred) in zip(uids, all_insts)}    # uid: pred mapping
        scores = evaluator.evaluate(uid2pred)
        # print(len(evaluator.uid2ref))
        scores['word_accu'] = word_accu

        return scores, uid2pred

    def save(self, name):
        encoder_path = os.path.join(self.output, '%s_enc.pth' % name)
        decoder_path = os.path.join(self.output, '%s_dec.pth' % name)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path):
        print("Load Speaker from %s" % path)
        enc_path = os.path.join(path + "_enc.pth")
        dec_path = os.path.join(path + "_dec.pth")
        enc_state_dict = torch.load(enc_path)
        dec_state_dict = torch.load(dec_path)
        self.encoder.load_state_dict(enc_state_dict)
        self.decoder.load_state_dict(dec_state_dict)
