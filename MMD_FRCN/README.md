## Finetuning

```
# transfer learning
__C.Trade_Off = 0.25 
__C.Target_Weight = 1
__C.TRANSFER_SELECT = 'POSITIVE' #CONDITION OR RANDOM OR POSITIVE
__C.TRANSFER_LOSS = 'MMD' #JMMD
```

## Data Preparation

Put your source data into Pascal Foramtted folder with folder name VOC2007 and link with data/VOCdevkit2007, put your target data into VOC2006 and link data/VOCdevkit2006 and follow the VOC training from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
