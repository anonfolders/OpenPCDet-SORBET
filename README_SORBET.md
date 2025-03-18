<!-- # Setup & Tasks
https://medium.com/@alifyafebriana/setting-up-3d-open-source-openpcdet-with-anaconda-a-step-by-step-guide-66126107215
1. download the model from https://github.com/open-mmlab/OpenPCDet/tree/master#model-zoo to /home/zhy/Desktop/PointPillar/OpenPCDet (copy)/checkpoints
2. change cmond in format python test.py --cfg_file /home/zhy/Desktop/PointPillar/OpenPCDet/tools/cfgs/kitti_models/ModelXXX.yaml --batch_size 10 --ckpt /home/zhy/Desktop/PointPillar/OpenPCDet/checkpoints/pcHECKpOINTxxx.pth
3. RUN,record the result

1. name->type
2. occluded->occlusion
3. alpha
4. dimensions
5. location
6. rotation_y
7. truncated->truncation
8. bbox->2D bbox
9. score->score

need to get detections with frame id for 3 models as the baseline
each model 3 times
redo the samething with the downloaded data.
-->


# 1.How to use (testing only)
1. conda activate PointPillarFor3080
2. put the dataset in data/kitti/training(\<root>/OpenPCDet_SORBET/data/kitti/training) and rename it to velodyne
3. if you want get the prediction without evaluation, excute command
```bash
python test.py --cfg_file <tools/cfgs/kitti_models/XXX.yaml> --batch_size 10 --ckpt <../checkpoints/YYY.pth> --raw_data ZZZ --round WWW
```
XXX is the configuration file, 

YYY is the pre-trained model you choose, 

ZZZ is the diretory to store the result, chosen from [point_pillar, point_rcnn, voxel_rcnn]. Output is in SORBET_output/raw_Data/ZZZ/WWW.

4. else if you want to get the evaluation, use this command and refer to the official reposite in GitHub
```
python test.py --cfg_file <root>/OpenPCDet/tools/cfgs/kitti_models/XXX.yaml --batch_size 10 --ckpt <root>/OpenPCDet/checkpoints/YYY.pth
```
5. To run a series of datasets, put the dataset in data/kitti/training and rename it to velodyne, name them as name_letter, and go to **tools/run_pointpillar.sh** in dir ```tools```. change the first line for data in ```data/kitti/training/*x``` to ...*letter you choose **For this mode, output is in  
```
<root>/OpenPCDet_SORBET/<.>/raw_res/[point_pillar,point_rcnn,voxel_rcnn]/<root>/OpenPCDet_SORBET/data/kitti/training/
```

# Modifications to OpenPCDet

## In ```tools/test.py```
1. Line 44-45: added arguments for ```test.py```
2. Line 61: added a function ```raw_res_one_epoch```
3. Line 225: added branch to get prediction without evaluation

## In ```tools/eval_utils```
1. added a  new file ```raw_data_utils.py``` which is simmilar to original ```eval_utils.py```
2. in **raw_data_utils.py**, from line 140, it is made for get the raw prediction. And it has not evauation part compared with **eval_utils.py**
3. in line 185,detections it's what I added to get the GT of the sample we use for preiction.

## in pcdet/datasets/kitti/kitti_dataset.py
in line 361,  added return value detections, to get the GT, and you can get the original GT by "which function gives me detection?"
## 