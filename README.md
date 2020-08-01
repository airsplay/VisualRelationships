# Data and Code for Paper "Expressing Visual Relationships via Language"
["Expressing Visual Relationships via Language"](https://arxiv.org/pdf/1906.07689.pdf), ACL 2019.

Hao Tan, Franck Dernoncourt, Zhe Lin, Trung Bui, and Mohit Bansal 

The slide of our talk in ACL 2019 is also available [here](http://www.cs.unc.edu/~airsplay/Hao_ACL_2019_slide.pdf).

## Image Editing Corpus Dataset
Json files in folder `data/` are three splits of the `ImgEdit` dataset. 
Descriptions of each field:
```
img0: The name of image to be edited.
img1: The name of image after editing.
sents: A list of image editing requests. One for training and three for valid / test.
uid: Universal id for each instruction. It is useful when multiple datasets are involved.
```

Image pairs are now available on google drive. [link to download](https://drive.google.com/drive/folders/1p_hkPwRUiLl1RHV3DkzQk3ti-GzHzT7O?usp=sharing).


## Existing Public Datasets
Besides our collected image-editing corpus, we also evalutae our methods on two public datasets in the paper:
- [nlvr2](https://arxiv.org/pdf/1811.00491.pdf): Cornell real image NLVR dataset; Github link: <https://github.com/clic-lab/nlvr/tree/master/nlvr2>
- [spotdiff](https://arxiv.org/pdf/1808.10584.pdf): the CMU spot-diff dataset; Github link: <https://github.com/harsh19/spot-the-diff>

Data are saved in the `dataset` folder and each subfolder is related to one dataset.
Each dataset contains `train.json`, `valid.json`, and `test.json` in the same format.

In general, each datum contains two images and (multiple) sentences to describe their relationships and are saved in json with format:
```
Data = [Datum]
Datum = {
    'img0': '/path/to/the-first-image',
    'img1': '/path/to/the-second-image',
    'sents': a list of reference sentences to describe the relationship between `img0` and `img1`
    'uid': The univesal id of this data
}, ...]
```

**More Related Datasets** 
1. The dataset [NEURAL NATURALIST](https://arxiv.org/abs/1909.04101) (published in EMNLP 2019) are created for a similar purpose. The task is to generate highlight differences between birds. Please take a look at their [website](https://mbforbes.github.io/neural-naturalist/#on-the-shoulders-of-Giants) and [Github]. 
2. The dataset [CLEVR-Change](https://arxiv.org/pdf/1901.02527.pdf) extends CLEVR with view-change captions. The [dataset](https://github.com/Seth-Park/viewpoint-invariant-change-captioning) was not publicly available yet. 

## Data Pre-processing

- Download the `nlvr2` and `spotdiff` datasets.
- Run pre-processing scripts in `dataset/`.
- Run vocab builders under `script/`. `build_vocab.py` would build the vocab from `train` and `valid` splits. (Please be careful here; the `test` split should always be excluded!)
- Run hdf5 builders under `script/`. `build_pixel_hdf5.py` would read the images, resize them, and save them in hdf5 files. `build_pixel_and_feat_hdf5.py` would extract the ResNet features and calculate the `mean` and `std` of features. (`test` splits have also been excluded from the calculation of `mean` and `std`)


## Model Training
The code should be run with `python3` and install: 
```
pip install torch torchvision h5py matplotlib Pillow numpy tensorboardX tqdm
```

You could run the model as simple as `bash run_speaker.sh`. More options are provided inside the script and I have annotated them.

Four model options are provided, which are corresponding to the four methods in our paper:

- `init`: Sec.3.1 Basic Model
- `newheads`: Sec.3.2 Sequential Multi-Head Attention
- `newcross`: Sec.3.3 Static Relational Attention
- `dynamic`: Sec.3.4 Dynamic Relational Attention


> Remind: If the model is `dynamic`, some of our GPUs have unknown precision issue: the results would dramatically decrease after a few epochs training. We are not sure about the reason but we found that our titan V cards do not suffer from this issue.


## Sequence-Level Training (Optional)
This code also provides the full utilization to train the model with reinforcement learning (RL), as shown in the paper [Sequence Level Training with Recurrent Neural Networks](https://arxiv.org/abs/1511.06732). It could be triggered by setting
```
--train rlspeaker
```
By setting this, the model would try to use the main metric (set by `--metric`) as reward and directly optimize it.
We also implement the `self-critical methods` shown in [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563), which could be activated by setting `--baseline self` in the running script.

The RL training could dramstically boost the optimized metric but show lower performance in human evaluation. We thus omitted RL training results in our paper (as discussed in Sec.5 Related Work) but the code is definitely free to use!

Enjoy it~

## Contact
Hao Tan: haotan@cs.unc.edu

Franck Dernoncourt: dernonco@adobe.com 

## Thanks
We specially thank [Nham Le](https://www.linkedin.com/in/nhamle/) in collecting the dataset!
