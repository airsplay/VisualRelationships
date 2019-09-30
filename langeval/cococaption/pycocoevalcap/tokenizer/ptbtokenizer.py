#!/usr/bin/env python
# 
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import sys
import subprocess
import tempfile
import itertools

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--", "...", ";"}
#PUNCTUATIONS = {}

class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def tokenize(self, captions_for_image):
        cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokenized_captions_for_image = {}
        temp_items = list(captions_for_image.items())
        image_id = [k for k, v in temp_items for _ in range(len(v))]
        sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in temp_items for c in v])

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences)
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(os.path.basename(tmp_file.name))
        FNULL = open(os.devnull, 'w')
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE, stderr=FNULL)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode("utf-8")
        lines = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, line in zip(image_id, lines):
            if not k in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                    if w not in PUNCTUATIONS])
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image

if __name__ == "__main__":
    tokenizer = PTBTokenizer()      # Tokenization
    gts = {}
    caps = ['Walk down the hall and wait in the doorway to the dining area. ', 'Walk past the bed and continue forwa rd, turning left at the first turn and stop at the door. ', 'Go forward past the bed. Turn right through door. Stop just inside door. ']
    caps = caps * 100
    gts[0] = [{'image_id': 0, 'id': k, 'caption': cap} for k, cap in enumerate(caps)]
    gts = tokenizer.tokenize(gts)
    print(gts)
