#!/usr/bin/env bash

WORK_DIR=$(pwd)

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=${WORK_DIR} python main.py \
                                +exps=moma_vid \
                                proj_name=deploy \
                                run_name=re_moma_vid \
                                trainer/wandb=wandb \
                                trainer.num_epochs=400 \
                                # checkpoint.save_ckpt=True \
