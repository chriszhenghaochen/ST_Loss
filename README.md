# Transferable Faster RCNN implemented with Pytorch

## Install and Setup

follow [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

## Finetuning
```
# transfer learning
__C.Trade_Off = 1
__C.TRANSFER_SELECT = 'CONDITION' #CONDITION OR RANDOM
__C.TRANSFER_LOSS = 'MMD' #JMMD or MMD
```
## Data Preparation

Put your source data into Pascal Foramtted folder with folder name VOC2007 and link with data/VOCdevkit2007, put your target data into VOC2006 and link data/VOCdevkit2006.

To change other data format (e.g. Kitti) to Pascal_VOC format, please refer to
[Kitti2Pascal](https://github.com/chriszhenghaochen/Kitti2Pascal) tutorial

## References
* *Ren, Shaoqing, et al. "Faster R-CNN: towards real-time object detection with region proposal networks." IEEE transactions on pattern analysis and machine intelligence 39.6 (2017): 1137-1149.*

* *Long, Mingsheng, et al. "Learning transferable features with deep adaptation networks." arXiv preprint arXiv:1502.02791 (2015).*

* *Long, Mingsheng, et al. "Deep transfer learning with joint adaptation networks." arXiv preprint arXiv:1605.06636 (2016).*


## Result
Will update result soon
