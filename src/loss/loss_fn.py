# from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Tuple

class MultipleTrainingObjectives(nn.Module):
    def __init__(
            self, 
            device,
            output_keys,
            loss_weights,
            loss_funcs,
            **kwargs
        ):
        super().__init__()

        self.device = device
        self.output_keys = output_keys
        self.loss_funcs = loss_funcs
        
        self.loss_weights = loss_weights
        self.train_losses = {
            **{'train_epoch_loss' : 0, 'train_step_loss' : 0},
            **{l_key : 0 for l_key in self.output_keys},
        }
        self.val_losses = {
            **{'val_epoch_loss' : 0, 'val_step_loss' : 0},
            **{l_key : 0 for l_key in self.output_keys}, 
        }
        
    def reset_per_epoch(self, is_eval : bool):
        if not is_eval:
            self.train_losses = {k : 0 for k, v in self.train_losses.items()}
        else:
            self.val_losses = {k : 0 for k, v in self.val_losses.items()}
            
    def reset_per_step(self, is_eval : bool):
        if not is_eval:
            self.train_losses = {k : (0 if k == 'train_step_loss' else v) for k, v in self.train_losses.items()}
        else:
            self.val_losses = {k : (0 if k == 'val_step_loss' else v) for k, v in self.val_losses.items()}
            
    def forward(
            self, 
            outputs, 
            data,
            is_eval : bool = False,
            **kwargs,
        ):
        self.reset_per_step(is_eval)
        loss_dict = self.train_losses if not is_eval else self.val_losses
        step_loss_key = 'train_step_loss' if not is_eval else 'val_step_loss'
        epoch_loss_key = 'train_epoch_loss' if not is_eval else 'val_epoch_loss'
        
        total_loss_cur_step = 0 
        for idx, (output_key, criterion) in enumerate(zip(self.output_keys, self.loss_funcs)):

            # key to retrieval
            target_key = output_key
            loss_key = output_key

            # retrieve current target, output
            output = outputs[output_key]
            target = data[target_key]
            weight = self.loss_weights[idx]

            # calculate loss corresponding to each training objective 
            if output_key == 'vid_action':
                loss = criterion(
                    output.to(self.device),
                    target.to(self.device)
                )
            
            elif output_key == 'comp_motion':
                B, MAX_OBJS_IN_BATCH, _ = output.shape

                output = output.view(B*MAX_OBJS_IN_BATCH, -1)
                target = target.view(B*MAX_OBJS_IN_BATCH, -1)
                output = output[target != -1].view(1, -1)
                target = target[target != -1].view(1, -1)

                loss = criterion(
                    output,
                    target.to(self.device)
                )
                
            elif output_key == 'comp_object':
                B, MAX_OBJS_IN_BATCH, D = output.shape

                output = output.view(B*MAX_OBJS_IN_BATCH, -1)
                target = target.view(B*MAX_OBJS_IN_BATCH, -1)
                output = output[target != -1].view(-1, D)
                target = target[target != -1].view(-1, D)
                target = target.max(dim=-1)[1]

                loss = criterion(
                    output,
                    target.to(self.device)
                )

            elif output_key == 'comp_TR':
                logits = output[0]
                target = output[1]

                logits = logits[target != -1].view(1, -1)
                target = target[target != -1].view(1, -1)

                if len(logits) == 0:
                    continue
                else:
                    loss = criterion(
                        logits,
                        target
                    )
            
            # weighted loss
            loss = loss * weight

            # put into loss dict
            loss_dict[loss_key] += loss # for visulaization

            # calculate total loss for current step
            total_loss_cur_step += loss # for backpropagation

        loss_dict[step_loss_key] = total_loss_cur_step      # for backpropagation
        loss_dict[epoch_loss_key] += total_loss_cur_step    # for visulaization
        
        return total_loss_cur_step, loss_dict