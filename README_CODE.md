# Data and Code for Paper "Expressing Visual Relationships via Language"
Data and Code for our paper "Expressing Visual Relationships via Language", Hao Tan, Franck Dernoncourt, Zhe Lin, Trung Bui, and Mohit Bansal, ACL 2019.


## Data
Data are saved in the `dataset` folder and each subfolder is related to one dataset.
Each dataset contains `train.json`, `valid.json`, and `test.json` in the same format.

In general, each datum contains two images and (multiple) sentences to describe their relationships.

Json Formats:

- `img0`: /path/to/the-first-image/on/adobe/server
- `img1`: /path/to/the-second-image/on/adobe/server
- `sents`: a list of reference sentences to describe the relationship between `img0` and `img1`
- `uid`: The univesal id of this data

Datasets:

- adobe: Our adobe dataset. One reference for train
- nlvr2: Cornell real image NLVR dataset. <https://github.com/clic-lab/nlvr/tree/master/nlvr2>
- spotdiff: the CMU spot-diff dataset. <https://github.com/harsh19/spot-the-diff>

Pre-processing:

- Download the `nlvr2` and `spotdiff` datasets.
- Run pre-processing scripts in `dataset/`.
- Run vocab builders under `script/`. `build_vocab.py` would build the vocab from `train` and `valid` splits. (Please be careful here; the `test` split should always be excluded!)
- Run hdf5 builders under `script/`. `build_pixel_hdf5.py` would read the images, resize them, and save them in hdf5 files. `build_pixel_and_feat_hdf5.py` would extract the ResNet features and calculate the `mean` and `std` of features. (`test` splits have also been excluded from the calculation of `mean` and `std`)


## Code
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

The RL training could dramstically boost the metric that it optimized but show a lower performance in human evaluation. We thus discard RL training in the paper (as discussed in Sec.5 Related Work) but it is definitely free to use!

Enjoy it~
