#!/usr/bin/env bash

WORK_DIR=$(pwd)

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=${WORK_DIR} python main.py \
                                +exps=cater_task1_comp \
                                trainer/wandb=wandb \
                                proj_name=deploy \
                                run_name=re_cater_task1_comp \
                                trainer.num_epochs=100 \
                                # checkpoint.save_ckpt=True \