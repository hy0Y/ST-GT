#!/usr/bin/env bash

WORK_DIR=$(pwd)

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

#############################
# trainer_mode = train_eval #
#############################

PYTHONPATH=${WORK_DIR} python main.py \
                                +exps=moma_vid \
                                trainer.num_epochs=1 \
#                                 dataset.num_workers=0 \

# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task1_comp \
#                                 trainer.num_epochs=1 \
#                                 dataset.num_workers=0 \

# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task1_vid \
#                                 trainer.num_epochs=1 \
#                                 dataset.num_workers=0 \

# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task2_comp \
#                                 trainer.num_epochs=1 \
#                                 dataset.num_workers=0 \

# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task2_vid \
#                                 trainer.num_epochs=1 \
#                                 dataset.num_workers=0 \
