# from __future__ import annotations

import os
import json
import torch
import logging
import numpy as np
import hydra

from tqdm import tqdm
# from typing import Tuple, List, Optional, Union

from utils.misc import compute_multiple_aps, sigmoid

class Trainer:
    def __init__(
            self, 
            cfg,
            model, 
            train_loader,
            val_loader,
            test_loader,
        ):
        self.cfg = cfg

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device(0)
        self.model = model.to(self.device)

        self.epochs = cfg.trainer.num_epochs
        
        # initialize loss function
        self.loss_funcs = hydra.utils.instantiate(
            cfg.trainer.training_objs,
            device=self.device
        )

        # initialize optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.trainer.optimizer, 
            params=model.parameters()
        )

        # initialize scheduler
        self.scheduler = hydra.utils.instantiate(
            config=cfg.trainer.scheduler,
            optimizer=self.optimizer, 
            steps_per_epoch=len(self.train_loader),
        )

        # initialize wandb
        self.wandb = hydra.utils.instantiate(
            cfg.trainer.wandb,
            model=self.model
        )
        
        # logging requirements
        if self.cfg.dataset.choice == 'cater': self.cater_load_mapping(log_dir=cfg.path.log_dir)
            
    def train_eval(self):

        for epoch in range(self.epochs):

            # training
            logging.info(f'epoch : {epoch} | Training')
            self.model.train()
            self.loss_funcs.reset_per_epoch(is_eval=False)            
            
            for idx, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='TRAIN', ncols=100, leave=True):
                
                # model forwarding
                outputs = self.model(data, is_eval=False)
                
                # cal loss
                train_step_loss, train_losses = self.loss_funcs(outputs, data['target'], is_eval=False, step=idx)
                
                # update
                self.optimizer.zero_grad()
                train_step_loss.backward()
                self.optimizer.step()

                # step scheduler
                if self.scheduler != None: self.scheduler.step()

            # retrieve current epoch loss for logging
            train_epoch_loss = train_losses['train_epoch_loss'].cpu().detach().item() / len(self.train_loader)
                        
            # save checkpoint
            self.save_checkpoint(epoch, train_epoch_loss)
            
            # validation
            with torch.no_grad():
                logging.info(f'epoch : {epoch} | Validation')
                self.model.eval()
                self.loss_funcs.reset_per_epoch(is_eval=True)
                y, y_h, y_s, val_ap = [], [], [], []
                for idx, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='VALID', ncols=100, leave=True):

                    # model forwarding
                    outputs = self.model(data, is_eval=True)
                    
                    # get loss
                    _, val_losses = self.loss_funcs(outputs, data['target'], is_eval=True)

                    pred_score = sigmoid(outputs['vid_action'].cpu().detach().numpy()[0])  # batch size => 1
                    
                    pred_label = (pred_score > 0.5)
                    true_label = data['target']['vid_action'].cpu().detach().numpy()[0]
                    num_classes = data['target']['vid_action'].shape[-1]

                    y.append(true_label)
                    y_h.append(pred_label)
                    y_s.append(pred_score)
                
                # retrieve current epoch loss for logging
                val_epoch_loss = val_losses['val_epoch_loss'].cpu().detach().item() / len(self.val_loader)

                val_ap = compute_multiple_aps(y, y_s)
            
            self.epoch_logging(
                self.cfg.dataset.choice,
                train_losses, 
                val_losses, 
                len(self.train_loader), 
                len(self.val_loader), 
                val_ap, 
                epoch
            )
        
    def test(self, ckpt_pth: str):
        
        # load checkpoint
        state_dict = torch.load(ckpt_pth)
        self.model.load_state_dict(state_dict["model_state_dict"])

        ######## test ########
        test_ap = []
        y, y_h, y_s = [], [], []
        with torch.no_grad():
            self.model.eval()
            for idx, data in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc='TEST', ncols=100, leave=True):

                # model forwarding
                outputs = self.model(data=data, is_eval=True)
                
                # get loss
                _, val_losses = self.loss_funcs(outputs, data['target'], is_eval=True)
                
                # prepare evaluation
                pred_score = sigmoid(outputs['vid_action'].cpu().detach().numpy()[0])  # batch size => 1
                pred_label = (pred_score > 0.5)
                true_label = data['target']['vid_action'].cpu().detach().numpy()[0]
                num_classes = data['target']['vid_action'].shape[-1]

                y.append(true_label)
                y_h.append(pred_label)
                y_s.append(pred_score)
                                    
            test_ap = compute_multiple_aps(y, y_s)
                
            logging.info('test ap per class :')
            logging.info(test_ap)
            logging.info(f'mAP : {test_ap[test_ap != -1.].mean()}')

    def save_checkpoint(
            self, 
            epoch : int, 
            loss : float
        ):
        if self.cfg.checkpoint.save_ckpt and \
            epoch >= self.cfg.checkpoint.save_min and \
            epoch % self.cfg.checkpoint.save_every == 0:

            os.makedirs(self.cfg.checkpoint.save_dir, exist_ok=True)

            checkpoint = dict()
            checkpoint['model_state_dict'] = self.model.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            checkpoint['loss'] = loss

            ckpt_name = 'ckpt_' + str(epoch) + '.pt'
            ckpt_dir = os.path.join(self.cfg.checkpoint.save_dir, ckpt_name)

            torch.save(checkpoint, ckpt_dir)

            logging.info(f'Epoch {epoch} | Training checkpoint saved to {ckpt_dir}')

    def epoch_logging(
            self,
            dtset: str,
            train_losses,
            val_losses,
            train_steps : int,
            val_steps : int,
            val_ap : np.ndarray,
            epoch : int
        ):

        train_logging_msg = {key : loss.cpu().detach().item() / train_steps  for key, loss in train_losses.items() if key != 'train_step_loss'}
        val_logging_msg = {key : loss.cpu().detach().item() / val_steps for key, loss in val_losses.items() if 'epoch' in key != 'val_step_loss'}
        
        
        logging.info('-------------------------------------------')
        logging.info(f'EPOCH : {epoch}')

        # train info logging
        logging.info('TRAIN LOSS')
        for msg, loss in train_logging_msg.items():
            logging.info(f'|__{msg} : {loss}')
        
        # val info logging
        logging.info('VALID LOSS')
        for msg, loss in val_logging_msg.items():
            logging.info(f'|__{msg} : {loss}')

        if dtset == 'cater':
            # per act logging
            if self.cfg.task == 'atomic':
                act2clsidxs = self.act2aaidxs
            else:
                act2clsidxs = self.act2caidxs
            
            act2ap = dict()
            for act_name, idxs in act2clsidxs.items():
                idxs = np.array(idxs)
                cur_act_aps = val_ap[idxs]
                logging.info(f'\"{act_name}\" Mean AP')
                logging.info(f"|__ {cur_act_aps[cur_act_aps != -1.].mean()}")
                act2ap.update({act_name : cur_act_aps[cur_act_aps != -1.].mean()})
        
        logging.info(f"lr : {self.optimizer.param_groups[0].get('lr')}")
        logging.info(f'mAP : {val_ap[val_ap != -1.].mean()}')
        logging.info('-------------------------------------------')
        
        if not self.wandb: return None

        if dtset == 'cater':
            self.wandb.log({
                **train_logging_msg,
                **val_logging_msg,
                **act2ap,
                **{'lr' : self.optimizer.param_groups[0]['lr']},
                **{'mAP' : val_ap[val_ap != -1.].mean()}
            }, step=epoch)
        else:
            self.wandb.log({
                **{'TRAIN EPOCH LOSS' : train_losses['train_epoch_loss'],
                   'VALID EPOCH LOSS' : val_losses['val_epoch_loss'],
                   'mAP'              : val_ap[val_ap != -1.].mean(),
                   'lr'               : self.optimizer.param_groups[0]['lr'],
            }}, step=epoch)    

    def cater_load_mapping(self, log_dir):
        aa_map_fname = 'AA_map.json'
        aa_map_pth = os.path.join(log_dir, aa_map_fname)
        ca_map_fname = 'CA_map.json'
        ca_map_pth = os.path.join(log_dir, ca_map_fname)
        act_map_fname = 'act_map.json'
        act_map_pth = os.path.join(log_dir, act_map_fname)

        with open(aa_map_pth, 'r') as j:
            aa_map = json.load(j)

        with open(ca_map_pth, 'r') as j:
            ca_map = json.load(j)
        
        with open(act_map_pth, 'r') as j:
            act_map = json.load(j)

        del act_map['None']

        act_names = list(act_map.keys())
        tr_names = ['before', 'during', 'after']
        act2aaidxs = {act : [] for act, _ in act_map.items()}
        act2caidxs = {act : [] for act in act_names + tr_names}

        for aa_name, aa_idx in aa_map.items():
            act_name = '_'.join(aa_name.split('_')[1:])
            act2aaidxs[act_name].append(aa_idx)
        
        for ca_name, ca_idx in ca_map.items():
            for act_name in act_names:
                if act_name in ca_name:
                    act2caidxs[act_name].append(ca_idx)
            for tr_name in tr_names:
                if tr_name in ca_name:
                    act2caidxs[tr_name].append(ca_idx)
        
        self.act_names = act_names
        self.act2aaidxs = act2aaidxs
        self.act2caidxs = act2caidxs