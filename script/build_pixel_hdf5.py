import os
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import shutil
import json

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import h5py
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
            img = img.convert('RGB')
        except Warning:
            print("A warning happens with", path)
        return img

resnet_extractor = models.resnet101(pretrained=True)
modules = list(resnet_extractor.children())[:-2]
resnet_extractor = nn.Sequential(*modules).cuda()
for p in resnet_extractor.parameters():
    p.requires_grad = False

DATASETS = [
        'nlvr2',
        'spotdiff',
        'adobe',
]

ds_root = "../dataset/"
for ds_name in DATASETS:
    print("Processing dataset %s" % ds_name)

    for split_name in [
            'train',
            'valid',
            'test',
            ]:
        data = []
        data.extend(
            json.load(open(os.path.join(ds_root, ds_name, split_name+".json")))
        )
        print("Finish Loading split %s" % split_name)

        img_raw = {'img0': [], 'img1': []}
        img_feat = {'img0': [], 'img1': []}

        for datum in tqdm(data):
            for key in ['img0', 'img1']:
                img_path = datum[key]
                img_processed_pixels = img_transform(pil_loader(img_path))  # 3 x 224 x 224 in cpu Tensor
                img_raw[key].append(img_processed_pixels.numpy())
        
        # Save the raw pixels
        pixel_hdf5_path = os.path.join(ds_root, ds_name, "%s_pixels.hdf5" % split_name)
        with h5py.File(pixel_hdf5_path, "w") as f:
            for key in img_raw:
                print("Stacking %s" % key)
                img_raw[key] = np.stack(img_raw[key])
                print(img_raw[key][-1, 0, 0, :5])
                print("Write %s" % key)
                dset = f.create_dataset(key, data=img_raw[key])
        print()

        # Test Reading
        with h5py.File(pixel_hdf5_path, "r") as f:
            for key in img_raw:
                print("test read %s" % key, f[key][-1, 0, 0, :5])
        print()
                
