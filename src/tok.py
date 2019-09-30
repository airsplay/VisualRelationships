from collections import defaultdict
import re
import numpy as np
import string

MIN_WORD_OCCURRENCE = 3

class Tokenizer:
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    def __init__(self):
        self.clear()
        self.add_speical_words()

    def clear(self):
        self._vocab_size = 0
        self._idx2word = {}
        self._word2idx = {}
        self.unk_idx = 0

    def add_speical_words(self):
        for word in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
            self.add_word(word)

    def add_word(self, word):
        self._idx2word[self._vocab_size] = word
        self._word2idx[word] = self._vocab_size
        self._vocab_size += 1
        return self._vocab_size

    def tokenize(self, sent):
        toks = []
        for word in Tokenizer.SENTENCE_SPLIT_REGEX.split(sent.strip()):
            word = word.strip().lower()
            if len(word) > 0:
                if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                    toks += list(word)
                else:
                    toks.append(word)
        return toks

    def build_vocab(self, sents, min_occur=MIN_WORD_OCCURRENCE):
        # Count
        self.occur = defaultdict(lambda: 0)
        for sent in sents:
            words = self.tokenize(sent)
            for word in words:
                self.occur[word.lower()] += 1

        # To have a meaningful dictionary, the word_id is sorted by its occurance
        # Pruning rare words
        wordXnum = sorted(self.occur.items(), key=lambda x:x[1], reverse=True)
        for word, num in wordXnum:
            if num >= min_occur:
                self.add_word(word)

    def word2idx(self, word, allow_unk=True):
        if word in self._word2idx:
            return self._word2idx[word]
        elif allow_unk:
            return self._word2idx['<UNK>']
        else:
            assert False, "No Word %s\n" % word

    def idx2word(self, idx):
        return self._idx2word[int(idx)]     # add int for handling tensors

    @property
    def vocab_size(self):
        return len(self._word2idx)

    @property
    def pad_id(self):
        return self.word2idx("<PAD>", allow_unk=False)

    @property
    def bos_id(self):
        return self.word2idx("<BOS>", allow_unk=False)

    @property
    def eos_id(self):
        return self.word2idx("<EOS>", allow_unk=False)

    @property
    def unk_id(self):
        return self.word2idx("<UNK>")

    def encode(self, sent):
        words = self.tokenize(sent)
        return list(map(lambda word: self.word2idx(word), words))

    def decode(self, idx):
        return " ".join(list(map(lambda i: self.idx2word(i), idx)))

    def dump(self, path):
        with open(path, 'w') as f:
            for i in range(len(self._idx2word)):
                f.write(self._idx2word[i] + "\n")

    def load(self, path):
        self.clear()
        with open(path, 'r') as f:
            for line in f:
                self.add_word(line.rstrip())

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word2idx('<EOS>', False))     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word2idx('<BOS>', False):
            start = 1
        else:
            start = 0
        return inst[start: end]

