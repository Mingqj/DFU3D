#!/usr/bin/env bash

set -x
NGPUS=$2
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/kitti_models/centerpoint.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/waymo_models/pointpillar_1x.yaml

# nuscenes
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/kitti_models/centerpoint_nuscenes2kitti.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/centerpoint_nuscenes2kitti/default_lr0.003_car60_truck26_bus57_ped31/ckpt/checkpoint_epoch_2.pth"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/kitti_models/pointrcnn_nuscenes2kitti.yaml


