from __future__ import annotations

import os

import torch
import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf
from typing import Tuple, List

from utils.misc import *
from model.basemodel import BaseModel
from dataset import get_loaders
from trainer import Trainer

def inital_setup(cfg : DictConfig) -> DictConfig:

    # hydra config resolver
    OmegaConf.register_new_resolver('sum_all', lambda *items : sum(items))
    OmegaConf.register_new_resolver('mul_all', lambda items: int(np.prod(items)))

    # seed setup
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
        
    assert cfg.trainer_mode in ['train_eval', 'test'], 'XX'
    assert cfg.test.model_ckpt_pth != None if cfg.trainer_mode == 'test' else True, 'XX'

    return cfg

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    # initial setup
    cfg = inital_setup(cfg)

    # data
    loaders = get_loaders(cfg, cfg.trainer_mode)

    # model initialization
    model = BaseModel(cfg)

    # init trainer
    trainer = Trainer(cfg, model, *loaders)

    # train
    if cfg.trainer_mode == 'train_eval':
        trainer.train_eval()
    else :
        trainer.test(cfg.test.model_ckpt_pth)

if __name__ == "__main__":
    main()