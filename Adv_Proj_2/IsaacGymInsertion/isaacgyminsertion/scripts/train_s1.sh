#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-test}
#NUM_ENVS=${4:-4096}
NUM_ENVS=${4:-4}
HEADLESS=${5:-False}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}

EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

#torchrun --standalone --nnodes=1 --nproc_per_node=3 \

CUDA_VISIBLE_DEVICES=0 /common/home/im316/miniforge3/envs/Advanced_Project2/bin/python train.py task=FactoryTaskInsertionTactile headless=False seed=${SEED} wandb_activate=True wandb_entity=ismarougkas wandb_project=Ismarou_Osher_Test multi_gpu=False restore_train=False task.grasp_at_init=False task.reset_at_fails=True task.reset_at_success=False task.env.numEnvs=${NUM_ENVS} task.env.compute_contact_gt=False task.external_cam.external_cam=False train.ppo.only_contact=False train.algo=PPO train.ppo.priv_info=True train.ppo.output_name="${CACHE}" ${EXTRA_ARGS}
