import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
import hydra

from omegaconf import DictConfig

from model.modules.classifiers import *

class BaseModel(nn.Module):
    '''
        BaseModel class is composed of following parts
            1. graph tokenizer
            2. graph transformer
            3. classifiers
    '''
    def __init__(
            self, 
            cfg : DictConfig
        ):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(0)
        
        # graph tokenizer
        self.graph_tokenizer = hydra.utils.instantiate(
            cfg.model.graph_tokenizer,
            device=self.device
        )

        # graph transformer
        self.graph_transformer = hydra.utils.instantiate(
            cfg.model.graph_transformer,
            device=self.device
        )
        
        # classifier
        self._classifier = hydra.utils.instantiate(
            cfg.model.classifiers,
            device=self.device
        )

    def forward(self, data, **kwargs):
        outputs = dict()
        vid_repr = self.graph_tokenizer(data)
        outputs = self.graph_transformer(vid_repr, data, outputs)
        outputs = self.classifier(outputs, data, kwargs['is_eval'])
        return outputs

    def classifier(self, outputs, data, is_eval):
        classifier = self._classifier

        # exps: cater_task1_comp.yaml
        if isinstance(classifier, EDM_task1):
            logits = classifier(
                inp=outputs['transformer_output'], 
                objmask=outputs['obj_mask'], 
                AAidxs_tgts=data['target']['edm_aa_idx'],
                is_eval=is_eval
            )
            outputs.update({
                f'vid_action'  : logits['final_outputs'],
                f'comp_motion' : logits['perobj_action'],
                f'comp_object' : logits['perobj_object'],
            })
        
        # exps: cater_task2_comp.yaml
        elif isinstance(classifier, EDM_task2):
            logits = classifier(
                inp=outputs['transformer_output'], 
                objmask=outputs['obj_mask'], 
                AAidxs_tgts=data['target']['edm_aa_idx'],
                TR_tgts=data['target']['comp_TR'],
                is_eval=is_eval
            )
            outputs.update({
                f'vid_action'      : logits['final_outputs'],
                f'comp_motion'     : logits['perobj_action'],
                f'comp_object'     : logits['perobj_object'],
                f'comp_TR'         : [logits['TR_logit'], logits['TR_target']],
            })
        
        # exps: cater_task1_vid.yaml or cater_task2_vid.yaml
        elif isinstance(classifier, MeanPoolMLP):
            logits = classifier(
                inp=outputs['transformer_output'], 
                objmask=outputs['obj_mask'] 
            )
            outputs.update({
                f'vid_action' : logits['final_outputs'],
            })

        # exps: moma_vid.yaml
        else: 
            logits = classifier(outputs['transformer_output'].mean(dim=1))
            outputs.update({f'vid_action' : logits})

        del(outputs['transformer_output'])
        return outputs
    