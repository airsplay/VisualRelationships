import random
import numpy as np

import torch
import torchvision.transforms as transforms

from param import args
from speaker import Speaker
from listener import Listener
from data import DiffDataset, TorchDataset

# Set the seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Image Transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
    normalize
])

# Change workers
if args.img_type == 'pixel':
    args.workers = 1    # In Memory Loading
elif args.img_type == 'feat':
    args.workers = 2

# Loading Dataset
def get_tuple(ds_name, split, task='speaker', shuffle=True, drop_last=True):
    dataset = DiffDataset(ds_name, split, args.train)
    torch_ds = TorchDataset(dataset, task, max_length=args.max_input,
        img0_transform=img_transform, img1_transform=img_transform
    )
    # if args.fast:
    #     torch_ds = torch.utils.data.Subset(torch_ds, range(1000))
    print("The size of data split %s is %d" % (split, len(torch_ds)))
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True,
        drop_last=drop_last)
    return dataset, torch_ds, loader

if 'speaker' in args.train:
    train_tuple = get_tuple(args.dataset, 'train', shuffle=False, drop_last=True)
    valid_tuple = get_tuple(args.dataset, 'valid', shuffle=False, drop_last=False)
    speaker = Speaker(train_tuple[0])   # [0] is the dataset
    if args.load is not None:
        print("Load speaker from %s." % args.load)
        speaker.load(args.load)
        scores, result = speaker.evaluate(valid_tuple)
        print("Have result for %d data" % len(result))
        print("The validation result is:")
        print(scores)
    if args.train == 'speaker':
        speaker.train(train_tuple, valid_tuple, args.epochs)
    if args.train == 'rlspeaker':
        speaker.train(train_tuple, valid_tuple, args.epochs, rl=True)
    elif args.train == 'validspeaker':
        scores, result = speaker.evaluate(valid_tuple)
        print(scores)
    elif args.train == 'testspeaker':
        test_tuple = get_tuple(args.dataset, 'test', shuffle=False, drop_last=False)
        scores, result = speaker.evaluate(test_tuple)
        print("Test:")
        print("Have result for %d data" % len(result))
        print(scores)
        import json
        json.dump(result, open("test_result.json", 'w'))
elif 'nlvr' in args.train:
    train_tuple = get_tuple(args.dataset, 'train', task='nlvr', shuffle=False, drop_last=True)
    valid_tuple = get_tuple(args.dataset, 'valid', task='nlvr', shuffle=False, drop_last=False)
    listener = Listener(train_tuple[0])
    if args.load is not None:
        listener.load(args.load)
    if args.train == 'nlvr':
        listener.train(train_tuple, valid_tuple, args.epochs)
    elif args.train == 'validnlvr':
        listener.evaluate(valid_tuple)
    elif args.train == 'snlvr':
        true_speaker = Speaker(train_tuple[0])   # [0] is the dataset
        if args.true_speaker is not None:
            print("Load speaker from %s." % args.load)
            true_speaker.load(args.true_speaker)
        listener = Listener(train_tuple[0], true_speaker=true_speaker)
        listener.speaker_evaluate(valid_tuple)
