#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-teacher}
NUM_ENVS=${4:-16}
HEADLESS=${5:-False}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth

CUDA_VISIBLE_DEVICES=0 python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} test=False task.data_logger.collect_data=True task.grasp_at_init=True task.reset_at_success=True task.reset_at_fails=True task.env.numEnvs=${NUM_ENVS} task.env.tactile=False task.external_cam.external_cam=True train.ppo.priv_info=True train.ppo.obs_info=True train.ppo.img_info=True train.ppo.seg_info=True train.ppo.pcl_info=False task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" train.algo=PPO ${EXTRA_ARGS}
