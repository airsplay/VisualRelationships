__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from functools import reduce

class COCOEvilCap:
    def __init__(self):
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self, gtscap, rescap, metrics=None, no_metrics=None):
        """

        :param gtscap: The gts of result. Support two types. [[gt1, gt2, ..., gt_n], ...] or [gt1, gt2, ...gt_n]
        :param rescap: The result. should be [res1, res2, ... res_n]
        :param metrics: 'BLEU', 'METEOR', 'ROUGE_L', 'CIDEr', 'Bleu_1'~'Bleu_4', 'F1'
        :param no_metrics": Exclude the metrics
        :return:
        """
        # self.eval = {}
        # self.imgToEval = {}
        # Preapre the sentence
        assert len(gtscap) == len(rescap)
        gts = {}
        res = {}
        if type(gtscap[0]) is list:
            cnt = 0
            for i, caps in enumerate(gtscap):
                gts[i] = [{'image_id': i, 'id': cnt + k, 'caption': cap} for k, cap in enumerate(caps)]
                cnt += len(caps)
        else:
            for i, cap in enumerate(gtscap):
                gts[i] = [{'image_id': i, 'id': i, 'caption': cap}]

        for i, cap in enumerate(rescap):
            res[i] = [{'image_id': i, 'id': i, 'caption': cap}]

        # =================================================
        # Set up scorers
        # =================================================
        tokenizer = PTBTokenizer()      # Tokenization
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        assert len(gts) == len(res)

        # =================================================
        # Set up scorers
        # =================================================
        metric2scorer = {
            'BLEU':     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'METEOR':   (Meteor(), "METEOR"),
            'ROUGE_L':  (Rouge(), "ROUGE_L"),
            'CIDEr':    (Cider(), "CIDEr")
        }

        fetch2cal = {
            "Bleu_1":   ("BLEU",), "Bleu_2": ("BLEU", ), "Bleu_3": ("BLEU",), "Bleu_4": ("BLEU",),
            "METEOR":   ("METEOR",),
            "ROUGE_L":  ("ROUGE_L",),
            "CIDEr":    ("CIDEr",),
            "F1":       ("BLEU", "ROUGE_L"),
            "BLEU":     ("BLEU",),
        }

        # Include Metrics
        if metrics is None:         # Include all the metrics
            fetch_metrics = set(list(metric2scorer.keys()) + ['F1', 'Bleu_1', 'Bleu_2', 'Bleu_3', "Bleu_4"])
        else:                       # Only include the mentioned metrics
            fetch_metrics = set(metrics)

        # Exclude Metrics
        if no_metrics is not None:
            fetch_metrics = fetch_metrics.difference(set(no_metrics))

        cal_metrics = set(sum([list(fetch2cal[fetch]) for fetch in fetch_metrics], []))
        scorers = [metric2scorer[m] for m in cal_metrics]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, list(gts.keys()), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, list(gts.keys()), method)

        # Calculate the BLEU from Bleu's
        gt_keys = list(gts.keys())
        if "BLEU" in fetch_metrics or "F1" in fetch_metrics:
            bleus = []
            for key in gt_keys:
                value = self.imgToEval[key]
                scores = [value[name] for name in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']]
                bleus.append(reduce(lambda x, y: x*y, scores, 1.) ** (0.25))     # Geometric Mean
            #self.setEval(sum(bleus) / len(bleus), "BLEU")
            self.setEval(reduce(lambda x, y: x*y, 
                [self.eval[name] for name in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']],
                1.) ** 0.25, 
                "BLEU")
            self.setImgToEvalImgs(bleus, gt_keys, "BLEU")

        # Calculate F1 from BLEU and Rouge_L
        if "F1" in fetch_metrics:
            f1s = []
            for key in gt_keys:
                value = self.imgToEval[key]
                bleu, rouge_l = value["BLEU"], value['ROUGE_L']
                if (bleu + rouge_l) == 0.:
                    f1 = 0.
                else:
                    f1 = 2 * (bleu * rouge_l) / (bleu + rouge_l)
                f1s.append(f1)
            # print(f1s)
            self.setEval(sum(f1s) / len(f1s), "F1")
            self.setImgToEvalImgs(f1s, gt_keys, "F1")

        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in list(self.imgToEval.items())]
