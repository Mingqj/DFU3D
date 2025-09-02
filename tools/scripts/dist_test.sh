#!/usr/bin/env bash

# set -x
# NGPUS=$1
# PY_ARGS=${@:2}

# kitti
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/centerpoint.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/centerpoint/default/ckpt/checkpoint_epoch_51.pth" --save_to_file --launcher pytorch ${PY_ARGS}
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/second.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/second/second_refinement_61epoch/ckpt/checkpoint_epoch_61.pth"  --launcher pytorch ${PY_ARGS}
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/pointpillar/pointpillar_refinement_70epoch/ckpt/checkpoint_epoch_70.pth"  --launcher pytorch ${PY_ARGS}
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/PartA2.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/PartA2/PartA2_refinement_80best/ckpt/checkpoint_epoch_80.pth"  --launcher pytorch ${PY_ARGS}

# # pandaset
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/second_ps.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/second_ps/default/ckpt/checkpoint_epoch_60.pth" --save_to_file --launcher pytorch ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --ckpt /opt/data/private/codeN/OpenPCDet/output/kitti_models/pointrcnn_iou/default/ckpt/checkpoint_epoch_67.pth --launcher pytorch ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/waymo_models/pointpillar_1x.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/waymo_models/pointpillar_1x/default/ckpt/checkpoint_epoch_6.pth" --save_to_file --launcher pytorch ${PY_ARGS}

# nuscenes2kitti
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/centerpoint_nuscenes2kitti.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/centerpoint_nuscenes2kitti/default_refine2_pseudo_box_vehicle32.3_cyclist_15.3_pedes37.7/ckpt/checkpoint_epoch_20.pth"  --launcher pytorch ${PY_ARGS}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/centerpoint_nuscenes2kitti.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/centerpoint_nuscenes2kitti/default/ckpt/checkpoint_epoch_9.pth" --launcher pytorch ${PY_ARGS}

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py --cfg_file cfgs/kitti_models/pointrcnn_nuscenes2kitti.yaml --ckpt "/opt/data/private/codeN/OpenPCDet/output/kitti_models/pointrcnn_nuscenes2kitti/default/ckpt/checkpoint_epoch_40.pth"  --launcher pytorch ${PY_ARGS} --save_to_file
