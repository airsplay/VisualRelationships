from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
from param import args
from tok import Tokenizer
from utils import BufferLoader
import copy
from PIL import Image
import json
import random
import os
import numpy as np
import torch

DATA_ROOT = "dataset/"

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_saver(img, path):
    # print(img, path)
    with open(path, 'wb') as f:
        img.save(f)

DEBUG_FAST_NUMBER = 1000

class DiffDataset:
    def __init__(self, ds_name='nlvr2', split='train', task='ispeaker'):
        self.ds_name = ds_name
        self.split = split
        self.data = json.load(
            open(os.path.join(DATA_ROOT, self.ds_name, self.split + ".json"))
        )

        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab.txt"))

        self.feat_mean = np.load(os.path.join(DATA_ROOT, self.ds_name, 'feat_mean.npy'))
        self.feat_std = np.load(os.path.join(DATA_ROOT, self.ds_name, 'feat_std.npy'))


class TorchDataset(Dataset):
    def __init__(self, dataset, task='speaker', max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.task = task
        self.tok = dataset.tok
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform

        if args.img_type == 'img':
            pass
        elif args.img_type == 'pixel':
            f = h5py.File(os.path.join(DATA_ROOT, self.dataset.ds_name, 
                self.dataset.split + "_pixels.hdf5"), 'r')
            if args.fast:
                self.img0_pixels = f['img0'][:DEBUG_FAST_NUMBER]
                self.img1_pixels = f['img1'][:DEBUG_FAST_NUMBER]
            else:
                self.img0_pixels = f['img0']
                self.img1_pixels = f['img1']
                assert len(self.img0_pixels) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels),
                                                                                    len(self.dataset.data))
                assert len(self.img1_pixels) == len(self.dataset.data)
        elif args.img_type == 'feat':
            f = h5py.File(os.path.join(DATA_ROOT, self.dataset.ds_name, 
                self.dataset.split + "_feats.hdf5"), 'r')
            if args.fast:
                self.img0_feats = f['img0'][:DEBUG_FAST_NUMBER]
                self.img1_feats = f['img1'][:DEBUG_FAST_NUMBER]
            else:
                self.img0_feats = f['img0']
                self.img1_feats = f['img1']
                assert len(self.img0_feats) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels),
                                                                                    len(self.dataset.data))
                assert len(self.img1_feats) == len(self.dataset.data)
        else:
            assert False

        # Make sure that each datum contains only one sent
        self.train_data = []
        self.id2imgid = {}
        for i, datum in enumerate(self.dataset.data):
            if args.fast and i >= DEBUG_FAST_NUMBER:     # Because I only load the top 1000 images
                break
            for sent in datum['sents']:
                new_datum = datum.copy()
                new_datum.pop('sents')
                new_datum['sent'] = sent
                self.id2imgid[len(self.train_data)] = i     # 'cause multiple inst may refer to the same image
                if self.dataset.ds_name == 'nlvr2' and task == 'speaker':
                    if new_datum['label'] == 'True':
                        self.train_data.append(new_datum)
                else:
                    self.train_data.append(new_datum)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        datum = self.train_data[item]
        uid = datum['uid']
        
        # Load Image
        img_id = self.id2imgid[item]        # Because one image may correspond to multiple data
        if args.img_type == 'img':
            img0_path = datum['img0']
            img1_path = datum['img1']
            img0 = self.img0_trans(pil_loader(img0_path))   # 3 x 224 x 224
            img1 = self.img1_trans(pil_loader(img1_path))
        elif args.img_type == 'pixel':
            img0 = torch.from_numpy(self.img0_pixels[img_id])       # 3 x 224 x 224
            img1 = torch.from_numpy(self.img1_pixels[img_id])
        elif args.img_type == 'feat':
            img0 = torch.from_numpy(self.img0_feats[img_id])        # 7 x 7 x 2048
            img1 = torch.from_numpy(self.img1_feats[img_id])
        else:
            assert False

        # Lang: Padding, <BOS>, <EOS> adding, cut the max
        sent = datum['sent']
        inst = self.tok.encode(sent)
        length = len(inst)
        a = np.ones((self.max_length), np.int64) * self.tok.pad_id
        a[0] = self.tok.bos_id
        if length + 2 < self.max_length:        # len( <BOS> + inst + <EOS> ) < max_len
            a[1: length+1] = inst
            a[length+1] = self.tok.eos_id
            length = 2 + length
        else:                                           # Else, clip the inst
            a[1: -1] = inst[:self.max_length-2]
            a[self.max_length-1] = self.tok.eos_id      # Force Ending
            length = self.max_length

        # Lang: numpy --> torch
        inst = torch.from_numpy(a)
        leng = torch.tensor(length)

        if self.task == 'speaker':
            return uid, img0, img1, inst, leng
        elif self.task == 'nlvr':
            if datum['label'] == 'True':
                label = 1.
            elif datum['label'] == 'False':
                label = 0.
            else:
                assert False
            return uid, img0, img1, inst, leng, label
