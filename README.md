# simNet
Implementation of "simNet: Stepwise Image-Topic Merging Network for Generating Detailed and Comprehensive Image Captions"

### Requirements
This code is written in Python2.7 and requires Torch 0.3

You need to download pre-trained Resnet152 model from [torchvision](https://github.com/pytorch/vision) for both training and evaluation. 

You can have a look at https://github.com/s-gupta/visual-concepts to get topic words
### Training a simNet model
Now we can train our simNet model with 

```
CUDA_VISIBLE_DEVICES=1,2,3 screen python train.py
```

### Testing a trained model
We can test our simNet model with 

```
CUDA_VISIBLE_DEVICES=1,2,3 screen python test.py
```
### Note
If you have any questions about the code or our paper, please send an email to "lfl@bupt.edu.cn"

### Reference
If you use this code as part of any published research, please acknowledge the following paper
```
@misc{Liu2018simNet,
author = {Fenglin Liu, Xuancheng Ren, Yuanxin Liu, Houfeng Wang, Xu Sun},
title = {simNet: Stepwise Image-Topic Merging Network for Generating Detailed and Comprehensive Image Captions},
journal = {EMNLP},
year = {2018}
}
```

### Acknowledgement 

Thanks to [Torch](http://torch.ch/) team for providing Torch 0.3, [CodaLab](https://competitions.codalab.org/) team for providing online evaluation，[COCO](http://cocodataset.org/) team and [Flickr30k](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/) for providing dataset, [Tsung-Yi Lin](https://github.com/tylin/coco-caption) for providing evaluation codes for MS COCO caption generation, [Yufeng Ma](https://github.com/yufengm)'s open source repositories and Torchvision [ResNet](https://github.com/pytorch/vision) implementation. 



