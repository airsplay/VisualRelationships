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


## Code
Please read [README_CODE](README_CODE.md) for more details.

## Contact
Hao Tan: haotan@cs.unc.edu

Franck Dernoncourt: dernonco@adobe.com 

## Thanks
We specially thank [Nham Le](https://www.linkedin.com/in/nhamle/) in collecting the dataset!
