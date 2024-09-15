#!/usr/bin/env bash

WORK_DIR=$(pwd)

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

#######################
# trainer_mode = test #
#######################

# # 70.81
# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=moma_vid \
#                                 trainer_mode=test \
#                                 test.model_ckpt_pth=/workspace/stgt_private/ckpt/re_moma_vid/ckpt_286.pt \

# # 99.73
# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task1_comp \
#                                 trainer_mode=test \
#                                 test.model_ckpt_pth=/workspace/stgt_private/ckpt/re_cater_task1_comp/ckpt_86.pt \

# # 90.58
# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task1_vid \
#                                 trainer_mode=test \
#                                 test.model_ckpt_pth=/workspace/stgt_private/ckpt/re_cater_task1_vid/ckpt_63.pt \

# # 75.46
# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task2_comp \
#                                 trainer_mode=test \
#                                 test.model_ckpt_pth=/workspace/stgt_private/ckpt/re_cater_task2_comp/ckpt_353.pt \

# # 61.26
# PYTHONPATH=${WORK_DIR} python main.py \
#                                 +exps=cater_task2_vid \
#                                 trainer_mode=test \
#                                 test.model_ckpt_pth=/workspace/stgt_private/ckpt/re_cater_task2_vid/ckpt_367.pt \