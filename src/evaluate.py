import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'langeval'))
from eval import LanguageEval

if __name__ == "__main__":
    dataset = json.load(open("../dataset/adobe/test.json"))
    langeval = LanguageEval()

    gts = []
    preds = []
    for datum in dataset:
        sents = datum['sents']
        for i in range(len(sents)):
            gt = sents[:i] + sents[i+1:]
            pred = sents[i]
            gts.append(gt)
            preds.append(pred)

    print(langeval.eval_whole(gts, preds))


class LangEvaluator():
    def __init__(self, dataset):
        self.uid2ref = {}
        self.langeval = LanguageEval()
        for datum in dataset.data:
            self.uid2ref[datum['uid']] = datum['sents'] 
            
    def evaluate(self, uid2pred):
        gts = []
        preds = []
        for uid, pred in uid2pred.items():
            preds.append(pred)
            gts.append(self.uid2ref[uid])

        return self.langeval.eval_whole(gts, preds, no_metrics={})

    def get_reward(self, uidXpred, metric="CIDEr"):
        gts = []
        preds = []
        for uid, pred in uidXpred:
            preds.append(pred)
            gts.append(self.uid2ref[uid])
        return self.langeval.eval_batch(gts, preds, metric)


if __name__ == "__main__":
    class D:
        def __init__(self):
            self.data = [{
                    "uid": "1",
                    "sents": ['I have a dream']
                    }]
    dataset = D()
                
    uid2pred = {"1": "I don't have a dream"}
    
    evaluator = LangEvaluator(dataset) 
    print(evaluator.evaluate(uid2pred))
    
