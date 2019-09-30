import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from tok import Tokenizer
import json

DATASETS = [
        'nlvr2',
        'spotdiff',
        'adobe',
]

ds_root = "../dataset/"
for ds_name in DATASETS:
    print("Processing dataset %s" % ds_name)

    dataset = []
    for split_name in ['train', 'valid']:
        dataset.extend(
            json.load(open(os.path.join(ds_root, ds_name, split_name+".json")))
        )
        print("Finish Loading split %s" % split_name)
    print("Number of data is %d." % len(dataset))
    sents = sum(map(lambda x: x["sents"], dataset), [])
    print("Number of sents is %d." % len(sents))

    tok = Tokenizer()
    tok.build_vocab(sents, min_occur=3)
    tok.dump(os.path.join(ds_root, ds_name, "vocab.txt"))

    wordXnum = list(tok.occur.items())
    wordXnum = sorted(wordXnum, key=lambda x:x[1], reverse=True)
    N = 50
    print("Top %d Words:" % N)
    for word, num in wordXnum[:N]:
        print("%s: %d" % (word, num))
    print()
    
    


