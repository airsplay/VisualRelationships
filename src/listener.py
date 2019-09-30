import torch
from time import time
import os
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from param import args
from model import ScoreModel, ConcatModel
import numpy as np
from evaluate import LangEvaluator
from utils import *

from tqdm import tqdm

class Listener:
    def __init__(self, dataset, true_speaker=None):
        # Built up the model
        self.tok = dataset.tok
        self.feature_size = 2048

        self.score_model = ScoreModel(dataset, self.feature_size, self.tok.vocab_size).cuda()

        self.true_speaker = true_speaker

        # Optimizer
        self.optim = args.optimizer(list(self.score_model.parameters()), lr=args.lr)

        # Logs
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output)     # Tensorboard summary writer

        # Loss
        self.bce_loss = torch.nn.BCEWithLogitsLoss(
            size_average = True, reduce = True
        )

    def train(self, train_tuple, eval_tuple, num_epochs):
        train_ds, train_tds, train_loader = train_tuple
        best_eval_score = -95

        avg_loss = 0.
        for epoch in range(num_epochs):
            iterator = tqdm(enumerate(train_loader), total=len(train_tds)//args.batch_size, unit="batch")
            #iterator = enumerate(train_loader)
            corrects = []
            start_time = time()
            for i, (uid, src, trg, inst, leng, label) in iterator:
                #print(src[0])
                #print(trg[0])
                inst = cut_inst_with_leng(inst, leng)
                src, trg, inst, label = src.cuda(), trg.cuda(), inst.cuda(), label.float().cuda()
                self.optim.zero_grad()
                loss, correct = self.teacher_forcing(src, trg, inst, leng, label, train=True)
                corrects.extend(correct.cpu().numpy())
                accu = sum(corrects) / len(corrects)
                if avg_loss == 0.:
                    avg_loss = loss.item()
                else:
                    alpha = 0.05
                    avg_loss = avg_loss * (1 - alpha) + alpha * loss.item()
                iterator.set_postfix(loss=avg_loss, accu=accu)
                loss.backward()
                nn.utils.clip_grad_norm_(self.score_model.parameters(), 5.)
                self.optim.step()
                if i % 300 == 0:
                    now_time = time()
                    time_taken = now_time - start_time
                    print("Train statistics at Epoch %d and Iter %d" % (epoch, i))
                    print("Avg Loss %0.4f" % avg_loss)
                    print("Accu of the currenct epoch till now is %0.4f" % accu)
                    print("Avg time %0.4f" % (time_taken / (i+1)))

            if epoch % 1 == 0:
                accu = self.evaluate(eval_tuple)
                log_str = "Epoch %d, accu %0.4f" % (epoch, accu)
                if accu > best_eval_score:
                    best_eval_score = accu
                    best_log_str = log_str
                    self.save("best_eval")
                print("BEST is %s" % best_log_str)
                print(log_str)
            print()

    def speaker_score(self, speaker, src, trg, inst, leng, train=True):
        """
        :param src:  src images (b x 224 x 224 x 3)?
        :param trg:  trg images
        :param inst: encoded sentences (b x max_len)
        :param leng: lengths of sentences (b)
        :param train: dropout or not?
        :return:
        """
        if train:
            speaker.encoder.train()
            speaker.encoder.resnet_extractor.eval()
            speaker.decoder.train()
        else:
            speaker.encoder.eval()
            speaker.decoder.eval()

        # Encoder
        ctx = speaker.encoder(src, trg)

        # Decoder
        batch_size = inst.size(0)
        h_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.hid_dim).cuda()
        logits, h1, c1 = speaker.decoder(inst, h_t, c_t, ctx, None)

        # softmax cross entropy loss
        softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tok.pad_id,
            reduction='none'
        )
        batch_size = logits.size(0)
        loss = softmax_loss(
            input  = logits[:, :-1, :].contiguous().view(-1, speaker.tok.vocab_size),         # -1 for aligning
            target = inst[:, 1:].contiguous().view(-1)              # "1:" to ignore the word <BOS>
        )
        loss = loss.view(batch_size, -1)
        # loss, _ = loss.max(1)
        loss = loss.sum(1) / (inst != speaker.tok.pad_id).cuda().sum(1).float()
        loss = torch.exp(loss)
        print(loss.size())
        print(loss)

        # Word Accuracy
        _, predict_words = logits.max(2)   # B, l
        correct = (predict_words[:, :-1] == inst[:, 1:])
        word_accu = correct.sum(1).float() / (inst != speaker.tok.pad_id).cuda().sum(1).float()
        print(word_accu)

        return word_accu

    def speaker_evaluate(self, eval_tuple):
        """
        Use the speaker to accomplish the listener task
        :param eval_tuple:
        :return:
        """
        dataset, th_dset, dataloader = eval_tuple

        # Generate sents by neural speaker
        self.score_model.eval()
        all_scores = []
        all_gts = []
        uids = []
        for i, (uid, src, trg, inst, leng, label) in tqdm(enumerate(dataloader)):
            inst = cut_inst_with_leng(inst, leng)
            src, trg, inst = src.cuda(), trg.cuda(), inst.cuda()
            loss = self.speaker_score(
                self.true_speaker, src, trg, inst, leng, train=False
            )
            all_scores.extend(loss.detach().cpu().numpy())
            all_gts.extend(label.numpy())
            uids.append(uid)
            # correct = 1.
            # for gt, pred in zip(all_gts, all_preds):
            #     if gt == pred:
            #         correct += 1
            # accu = correct / len(all_gts)
            # print(accu)
            # if i == 5:
            #     break

        # Fihnd Calculate accuracy
        all_scores = np.array(all_scores)
        max_accu = 0.
        max_threshold = 0.
        for threshold in np.linspace(0, 1, 100):
            correct = 0.
            all_preds =  all_scores < threshold
            for gt, pred in zip(all_gts, all_preds):
                if gt == pred:
                    correct += 1
            accu = correct / len(all_gts)
            print(threshold, accu)
            if accu > max_accu:
                max_accu = accu
                max_threshold = threshold

        print(max_threshold, max_accu)
        return accu

    def evaluate(self, eval_tuple):
        dataset, th_dset, dataloader = eval_tuple

        # Generate sents by neural speaker
        self.score_model.eval()
        all_preds = []
        all_gts = []
        uids = []
        for i, (uid, src, trg, inst, leng, label) in enumerate(dataloader):
            inst = cut_inst_with_leng(inst, leng)
            src, trg, inst = src.cuda(), trg.cuda(), inst.cuda()
            pred = self.infer_batch(src, trg, inst, leng, sampling=False, train=False)
            all_preds.extend(pred.cpu().numpy())
            all_gts.extend(label.numpy())
            uids.append(uid)

        # Calculate accuracy
        correct = 0.
        for gt, pred in zip(all_gts, all_preds):
            if gt == pred:
                correct += 1
        accu = correct / len(all_gts)

        return accu

    def teacher_forcing(self, src, trg, inst, leng, label, train=True):
        """
        :param src:  src images (b x 224 x 224 x 3)?
        :param trg:  trg images
        :param inst: encoded sentences (b x max_len)
        :param leng: lengths of sentences (b)
        :param train: dropout or not?
        :return:
        """
        if train:
            self.score_model.train()
        else:
            self.score_model.eval()

        logit = self.score_model(src, trg, inst, leng)
        pred = (logit > 0).float()
        correct = (pred == label)

        # softmax cross entropy loss
        loss = self.bce_loss(
            input=logit,
            target=label,
        )

        return loss, correct

    def infer_batch(self, src, trg, inst, leng, sampling=True, train=False):
        """
        :param src:  src images (b x 224 x 224 x 3)?
        :param trg:  trg images
        """
        self.score_model.eval()
        logit = self.score_model(src, trg, inst, leng)
        pred = (logit > 0).float()
        return pred

    def save(self, name):
        model_path = os.path.join(self.output, '%s.pth' % name)
        torch.save(self.score_model.state_dict(), model_path)

    def load(self, path):
        print("Load Speaker from %s" % path)
        state_dict = torch.load(path)
        self.score_model.load_state_dict(state_dict)
