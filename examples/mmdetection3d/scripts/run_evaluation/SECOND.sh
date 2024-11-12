#!/bin/bash

export MMDET3D_HOME="/home/yingqi/repo/mmdetection3d" && python ${MMDET3D_HOME}/tools/test.py ${MMDET3D_HOME}/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py /home/ioeddk/GitHub/torchsparse-dev/examples/mmdetection3d/pretrained_models/backup/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3-converted.pth --cfg-options test_evaluator.pklfile_prefix=outputs/torchsparse/second --cfg-options model.middle_encoder.type=SparseEncoderTS --task lidar_det