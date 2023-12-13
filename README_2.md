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