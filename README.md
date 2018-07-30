# Faster RCNN for multi-domains

Using selective loss, currently avaliable for pedestratin detection(binary class), 2 domains.

## Install and Setup

Follow [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

## Finetuning

```
#Transfer learning:
__C.TRANSFER = False            #if this is False, it woule be baseline, Faster RCNN
__C.TRANSFER_SELECT = 'ALL'     #select "ALL", "POSITIVE", "BALANCE" or "RANDOM"
__C.TRANSFER_WEIGHT = 0.5       #alpha
__C.TRANSFER_GAMMA = 16         #gamma
__C.TRANSFER_LOSS_START = 100   #transfer loss start epoch
__C.TRANSFER_GRL = False        #use Grandient Reversal or not
```


## Data Preparation
Put your target data into Pascal Foramtted folder with folder name VOC2007 and link with data/VOCdevkit2007, put your source data into VOC2012 and link data/VOCdevkit2012 and follow the VOC_0712 training from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

To change other data format (e.g. Kitti) to Pascal_VOC format, please refer to
[Kitti2Pascal](https://github.com/chriszhenghaochen/Kitti2Pascal) tutorial


## Result
Will update Result soon



