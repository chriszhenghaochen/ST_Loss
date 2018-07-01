# Transferable Faster RCNN implemented with Pytorch

## Install and Setup

Go to GRL_FRCN and MMD_FRCN then follow [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

## Finetuning

### MMD_FRCN

```
# transfer learning
__C.Trade_Off = 0.25 
__C.Target_Weight = 1
__C.TRANSFER_SELECT = 'POSITIVE' #CONDITION OR RANDOM OR POSITIVE
__C.TRANSFER_LOSS = 'MMD' #JMMD
```

### GRL_FRCN

```
# transfer learning
__C.TRANSFER = True
__C.TRANSFER_SELECT = 'POSITIVE'
__C.TRANSFER_WEIGHT = 1
__C.SOURCE_WEIGHT = 0.5
```

## Data Preparation

### MMD_FRCN
Put your source data into Pascal Foramtted folder with folder name VOC2007 and link with data/VOCdevkit2007, put your target data into VOC2006 and link data/VOCdevkit2006 and follow the VOC training from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

### GRL_FRCN
Put your source data into Pascal Foramtted folder with folder name VOC2007 and link with data/VOCdevkit2007, put your target data into VOC2012 and link data/VOCdevkit2012 and follow the VOC_0712 training from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

### VOC Format 
To change other data format (e.g. Kitti) to Pascal_VOC format, please refer to
[Kitti2Pascal](https://github.com/chriszhenghaochen/Kitti2Pascal) tutorial


## Result
**Data**: [ECCV Workshop WIDER FACE AND PEDESTRIAN CHALLENGE 2018](http://www.wider-challenge.org/) Challenge 2. Use Surveillance data set as **source** and Driven data set as **target**.

| Result           | Surveillance  | Driven |
| -----------------|:-------------:| -----: |
| Baseline (FRCN)  |   71.9        |  63.8  |
| MMD_FRCN         |   70.0        |  64.9  |
| GRL_FRCN         |   71.5        |  64.3  |


## References
* *Ren, Shaoqing, et al. "Faster R-CNN: towards real-time object detection with region proposal networks." IEEE transactions on pattern analysis and machine intelligence 39.6 (2017): 1137-1149.*

* *Long, Mingsheng, et al. "Learning transferable features with deep adaptation networks." arXiv preprint arXiv:1502.02791 (2015).*

* *Long, Mingsheng, et al. "Deep transfer learning with joint adaptation networks." arXiv preprint arXiv:1605.06636 (2016).*

* *Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17.1 (2016): 2096-2030.*

* *Chen, Yuhua, et al. "Domain adaptive faster r-cnn for object detection in the wild." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.*

