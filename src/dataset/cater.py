# from __future__ import annotations

import os
import logging
import numpy as np
import json
import torch

# from typing import List, Tuple, Dict
from omegaconf import DictConfig, OmegaConf

from torch.nn.utils.rnn import pad_sequence
from utils.misc import *

# token types : Node, Spatial Edge, Temporal Edge
TYPE_N = 0
TYPE_SE = 1
TYPE_TE = 2

NUM_CLIPS = 10
FPC = 30

class CaterDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            mode : str, 
            dataset_cfg : DictConfig,
            **kwargs
        ):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.dataset_cfg = dataset_cfg
        self.mode = mode
        self.task = dataset_cfg.task       

        # load prprocessed data (input)
        self.video_ids = self.get_video_ids(mode)

        # load prprocessed data (target)
        self.vid2target_vid_action = self.get_target_vid_action(
            task=self.task,
            vids=self.video_ids,
        )
        if self.dataset_cfg.use_edm:
            self.vid2target_perobj_object = self.get_target_perobj_object(vids=self.video_ids)
            self.vid2target_perobj_action = self.get_target_perobj_action(vids=self.video_ids)

    def get_video_ids(self, mode: str):
        logging.info(f'{mode} dataset : get video ids')

        split_pth = os.path.join(self.dataset_cfg.split_dir, self.task, f'{mode}_split.txt')
        with open(split_pth, 'r') as f:
            vids = [line.rstrip() for line in f.readlines()]
        return vids
    
    def get_metadatas(self, vid, ED, NC):
        mtdata_pth = os.path.join(self.dataset_cfg.metadata_dir, f'ED_{ED}_NC_{NC}', 'basic_info', f'{vid}.npy')
        metadata = np.load(mtdata_pth, allow_pickle=True).item()
        return metadata

    def get_target_vid_action(self, task, vids):
        target = 'task1.npy' if task == 'atomic' else 'task2.npy'
        target_pth = os.path.join(self.dataset_cfg.target_dir, target)
        vid2target_aas = np.load(target_pth, allow_pickle=True).item()
        return {vid : tgt for vid, tgt in vid2target_aas.items() if vid in vids}

    def get_target_perobj_action(self, vids):
        target_pth = os.path.join(self.dataset_cfg.target_dir, 'decomposed', 'semantic_el_act.npy')
        return {vid : tgt for vid, tgt in np.load(target_pth, allow_pickle=True).item().items() if vid in vids}

    def get_target_perobj_object(self, vids):
        target_pth = os.path.join(self.dataset_cfg.target_dir, 'decomposed', 'semantic_el_obj.npy')
        return {vid : tgt for vid, tgt in np.load(target_pth, allow_pickle=True).item().items() if vid in vids}

    def get_target_temporal_relation(self, vid):
        target_pth = f'{self.dataset_cfg.target_dir}/decomposed/semantic_el_tr/{vid}.npy'
        return np.load(target_pth, allow_pickle=True).item()

    def __len__(self):
        return len(self.video_ids)
        
    def __getitem__(self, idx):

        # get current video ID
        vid = self.video_ids[idx]

        # get preprocessed metadata
        metadata = self.get_metadatas(vid, self.dataset_cfg.edge_dist, self.dataset_cfg.num_clips)

        num_objs = metadata['num_inst']

        # token type : Spatial Edge, node idx pair
        Spatial_E_pair_idx = metadata['spatial_edge_idxs'] 

        # token type : Spatial Edge, time pair
        Spatial_E_pair_time = metadata['spatial_time_idxs']

        # token type : Node
        Node_pair_idx, Node_pair_time = [], []              # num_clip, total nodes in clip 

        # token type : Temporal Edge
        Temporal_E_pair_idx, Temporal_E_pair_time = [], []  # num_clip, (num_frame-1)*objs

        for clip_idx, (s, e) in enumerate(metadata['frame_info']):
            # current clips => s ~ e-1
            num_frames = e - s # 30

            # Token type : Node, node idx pair
            Node_pair_idx.append(
                [(obj_idx, obj_idx) for obj_idx in range(1, num_objs+1)] * num_frames
            )

            # Token type : Node, time pair
            cur_clip_node_time_idxs = []
            for ts in range(s, e):
                cur_clip_node_time_idxs.extend([(ts, ts) for _ in range(num_objs)])
            Node_pair_time.append(cur_clip_node_time_idxs)

            # Token type : Temporal Edge, node idx pair
            Temporal_E_pair_idx.append(
                [(obj_idx + 1, obj_idx + 1) for obj_idx in range(num_objs)] * (num_frames - 1)
            )

            # Token type : Temporal Edge, time pair
            cur_clip_temp_edge_time_idxs = []
            for ts in range(s, e-1):
                cur_clip_temp_edge_time_idxs.extend([(ts, ts+1) for _ in range(num_objs)])
            Temporal_E_pair_time.append(cur_clip_temp_edge_time_idxs)
        
        # aggregate 3 different type's tokens in one list
        # num_clips, T
        token_pair_idx, token_pair_time, token_types = [], [], []
        pad_mask = []
        for clip_idx in range(self.dataset_cfg.num_clips):
            token_pair_idx.append(
                torch.Tensor(
                    Node_pair_idx[clip_idx] + \
                    Spatial_E_pair_idx[clip_idx] + \
                    Temporal_E_pair_idx[clip_idx]
                )
            )
            token_pair_time.append(
                torch.Tensor(
                    Node_pair_time[clip_idx] + \
                    Spatial_E_pair_time[clip_idx] + \
                    Temporal_E_pair_time[clip_idx]
                )
            )
            token_types.append(
                torch.Tensor(
                    [TYPE_N] * len(Node_pair_idx[clip_idx]) + \
                    [TYPE_SE] * len(Spatial_E_pair_idx[clip_idx]) + \
                    [TYPE_TE] * len(Temporal_E_pair_idx[clip_idx])
                )
            )
            pad_mask.append(
                torch.Tensor(
                    [1] * len(Node_pair_idx[clip_idx] + Spatial_E_pair_idx[clip_idx] + Temporal_E_pair_idx[clip_idx])
                )
            )

        # get attr feature
        attr_feats = torch.from_numpy(np.load(f'{self.dataset_cfg.feat_dir}/{vid}.npy'))

        # get dof feature
        objbytsbycoord = torch.from_numpy(np.load(f'{self.dataset_cfg.dof_dir}/{vid}.npy'))  # 11, 301, 12
        coord_feat = []
        for clip_token_pair_idx, clip_token_pair_time in zip(token_pair_idx, token_pair_time):

            T = len(clip_token_pair_idx)
            coord_feat_clip = torch.zeros((T, 24))

            _1_n_idx = clip_token_pair_idx[:, 0].long()
            _2_n_idx = clip_token_pair_idx[:, 1].long()

            _1_t_idx = clip_token_pair_time[:, 0].long()
            _2_t_idx = clip_token_pair_time[:, 1].long()

            coord_feat_clip[:, :12] = objbytsbycoord[_1_n_idx, _1_t_idx]  # 지금 clip의 tokens, 12
            coord_feat_clip[:, 12:] = objbytsbycoord[_2_n_idx, _2_t_idx]  # 지금 clip의 tokens, 12

            coord_feat.append(coord_feat_clip)  # clip당  1, T(본인clip의 token 수), 24
        
        # node identifier's idx
        idx_in_lookup = []
        for clip_idx, (clip_token_pair_idx, clip_token_pair_time) in enumerate(zip(token_pair_idx, token_pair_time)):
            nidx_in_clip = torch.Tensor(clip_token_pair_idx) - 1             # 536, 2
            tidx_in_clip = torch.Tensor(clip_token_pair_time) % FPC     # 536, 2
            idx_in_adj_clip = nidx_in_clip + num_objs * tidx_in_clip
            idx_in_lookup.append(idx_in_adj_clip)


        '''
            As position encoding for graphs, two options are considered
                option 1: the eigenvectors of the graph Laplacian matrix
                option 2: the Q matrix from a random Gaussian matrix.

            ref: https://arxiv.org/pdf/2207.02505 
        '''
        # NOTE option 1 
        # # laplacian's eigen vectors [clip1's eigvec(270 x 270), clip2's eigvec, ... clip10's eigvec]
        # adj_pth = os.path.join(
        #     self.dataset_cfg.metadata_dir,
        #     f'ED_{self.dataset_cfg.edge_dist}_NC_{self.dataset_cfg.num_clips}',
        #     'adj_matrix',
        #     f'{vid}.npy'
        # )
        # adj_mats = torch.from_numpy(np.load(adj_pth)) # num_clips, #obj * ts, #obj * ts
        # n_ids = []
        # n_node = adj_mats.shape[1]
        # for clip_idx in range(adj_mats.shape[0]):
        #     adj_clip = adj_mats[clip_idx]
        #     in_degree_clip = adj_clip.long().sum(dim=1).view(-1)
        #     EigVec, _ = lap_eig(adj_clip, n_node, in_degree_clip)   # 각 row가 대응하는 eigvec
        #     n_ids.append(EigVec.unsqueeze(0))
        # n_ids = torch.cat(n_ids, dim=0)

        # NOTE option 2 
        RG = torch.randn(NUM_CLIPS, num_objs * FPC, num_objs * FPC)
        Q, _ = torch.linalg.qr(RG)
        n_ids = Q
        
        target_vid_action = self.vid2target_vid_action[vid]
        
        if self.dataset_cfg.use_edm:
            perobj_action = self.vid2target_perobj_action[vid]
            perobj_object = self.vid2target_perobj_object[vid]
            target_CGs = self.get_target_temporal_relation(vid)
            target_AAs = target_CGs['aa_label']
            target_TRs = target_CGs['temporal_label']

        token_pair_idx = pad_sequence(token_pair_idx, batch_first=True, padding_value=0)
        token_pair_time = pad_sequence(token_pair_time, batch_first=True, padding_value=0)
        token_types = pad_sequence(token_types, batch_first=True, padding_value=0)
        pad_mask = pad_sequence(pad_mask, batch_first=True, padding_value=0)
        coord_feat = pad_sequence(coord_feat, batch_first=True, padding_value=0)
        idx_in_lookup = pad_sequence(idx_in_lookup, batch_first=True, padding_value=0)

        data = {
            ### metadata    
            'num_objs' : num_objs,                  
            'tokens' :  token_pair_idx.shape[1],    # num_tokens

            ### inputs
            'token_pair_idx'  : token_pair_idx,     # torch.Tensor of shape [10, 651, 2]         -> per clip
            'token_pair_time' : token_pair_time,    # torch.Tensor of shape [10, 651, 2]         -> per clip
            'token_types'     : token_types,        # torch.Tensor of shape [10, 651]            -> per clip
            'pad_mask'        : pad_mask,           # torch.Tensor of shape [10, 651]            -> per clip

            'attr_feats'      : attr_feats,         # torch.Tensor of shape num_obj + 1, 200     -> per video
            'coord_feat'      : coord_feat,         # torch.Tensor of shape [10, 651, 24]        -> per clip

            'idx_in_lookup'      : idx_in_lookup,         # torch.Tensor of shape [10, 651, 2]         -> per clip
            # 'lap_eigens'      : lap_eigens,         # torch.Tensor of shape [10, 270, 270]       -> per clip
            'node_ids'        : n_ids,              # torch.Tensor of shape [10, 270, 270]       -> per clip

            ### targets
            'target_vid_action' : torch.Tensor(target_vid_action),
            'target_perobj_action' : torch.Tensor(perobj_action)  if self.dataset_cfg.use_edm else None,
            'target_perobj_object' : torch.Tensor(perobj_object)  if self.dataset_cfg.use_edm else None,
            'target_AAs' : torch.Tensor(target_AAs)               if self.dataset_cfg.use_edm else None,
            'target_TRs' : torch.Tensor(target_TRs)               if self.dataset_cfg.use_edm else None,
        }

        return data

    def custom_collate_fn(self, batch):
        B = len(batch)
        NC = self.dataset_cfg.num_clips
        MAX_TOKEN_LEN = max([x.shape[1] for x in [data['token_pair_idx'] for data in batch]])
        MAX_OBJS_IN_BATCH = max([data['num_objs'] for data in batch])

        # meta data
        b_num_objs = [data['num_objs'] for data in batch]
        b_tokens = [data['tokens'] for data in batch]

        # inputs
        b_token_pair_idx = [data['token_pair_idx'] for data in batch]
        b_token_pair_time = [data['token_pair_time'] for data in batch]
        b_token_types = [data['token_types'] for data in batch]
        b_pad_mask = [data['pad_mask'] for data in batch]

        b_attr_feats = [data['attr_feats'] for data in batch]
        b_coord_feat = [data['coord_feat'] for data in batch]

        b_idx_in_lookup = [data['idx_in_lookup'] for data in batch]
        b_node_ids = [data['node_ids'] for data in batch]

        # target
        b_tgt_vid_action_l = [data['target_vid_action'] for data in batch]
        if self.dataset_cfg.use_edm:       
            b_tgt_perobj_action = [data['target_perobj_action'] for data in batch] 
            b_tgt_perobj_object = [data['target_perobj_object'] for data in batch] 
            b_tgt_aas = [data['target_AAs'] for data in batch]
            b_tgt_trs = [data['target_TRs'] for data in batch]

        # pad to token dimension
        p_token_pair_idx = torch.zeros(B, NC, MAX_TOKEN_LEN, 2)
        p_token_pair_time = torch.zeros(B, NC, MAX_TOKEN_LEN, 2)
        p_token_types = torch.zeros(B, NC, MAX_TOKEN_LEN)
        p_pad_mask = torch.zeros(B, NC, MAX_TOKEN_LEN)
        p_coord_feat = torch.zeros(B, NC, MAX_TOKEN_LEN, 24)
        p_idx_in_lookup = torch.zeros(B, NC, MAX_TOKEN_LEN, 2)
        p_node_ids = torch.zeros(B, NC, 300, 300) # NOTE temporally 

        if self.dataset_cfg.use_edm:
            MAX_uniq_aas = max([tr.shape[0] for tr in b_tgt_trs])
            p_tgt_trs = torch.zeros(B, MAX_uniq_aas, MAX_uniq_aas, 3)

        for batch_idx in range(B):
            cur_len = b_tokens[batch_idx]
            rows = b_node_ids[batch_idx].shape[1]
            p_token_pair_idx[batch_idx, :NC, :cur_len, :] = b_token_pair_idx[batch_idx]
            p_token_pair_time[batch_idx, :NC, :cur_len, :] = b_token_pair_time[batch_idx]
            p_token_types[batch_idx, :NC, :cur_len] = b_token_types[batch_idx]
            p_pad_mask[batch_idx, :NC, :cur_len] = b_pad_mask[batch_idx]
            p_coord_feat[batch_idx, :NC, :cur_len, :] = b_coord_feat[batch_idx]
            p_idx_in_lookup[batch_idx, :NC, :cur_len, :] = b_idx_in_lookup[batch_idx]
            p_node_ids[batch_idx, :NC, :rows, :rows] = b_node_ids[batch_idx]

            if self.dataset_cfg.use_edm:
                num_uniq_aas = b_tgt_trs[batch_idx].shape[0]
                p_tgt_trs[batch_idx, :num_uniq_aas, :num_uniq_aas, :] = b_tgt_trs[batch_idx]
            
        p_attr_feats = pad_sequence(b_attr_feats, batch_first=True, padding_value=0)

        p_tgt_vid_action = pad_sequence(b_tgt_vid_action_l, batch_first=True, padding_value=0)

        if self.dataset_cfg.use_edm:
            p_tgt_perobj_action = pad_sequence(b_tgt_perobj_action, batch_first=True, padding_value=-1)
            p_tgt_perobj_object = pad_sequence(b_tgt_perobj_object, batch_first=True, padding_value=-1)
            p_tgt_aas = pad_sequence(b_tgt_aas, batch_first=True, padding_value=-1)

        obj_mask = torch.zeros((B, MAX_OBJS_IN_BATCH))
        for batch_idx in range(B):
            cur_num_objs = b_num_objs[batch_idx]
            obj_mask[batch_idx, :cur_num_objs] = 1
            obj_mask[batch_idx, cur_num_objs:] = 0

        # make node type token selection mask
        each_obj_bool = torch.zeros((B, MAX_OBJS_IN_BATCH, NC*MAX_TOKEN_LEN))        
        vid_tok_idxs = p_token_pair_idx.view(B, NC*MAX_TOKEN_LEN, 2)
        vid_tok_typs = p_token_types.view(B, NC*MAX_TOKEN_LEN)
        typ_0_bool = (vid_tok_typs[:, :] == 0)  
        for obj_idx in range(MAX_OBJS_IN_BATCH):
            obj_idx_in_map = obj_idx + 1
            cur_obj_src_bool = (vid_tok_idxs[:, :, 0] == obj_idx_in_map)
            cur_obj_des_bool = (vid_tok_idxs[:, :, 1] == obj_idx_in_map)
            node_and_temp_edge = torch.logical_and(cur_obj_src_bool, cur_obj_des_bool)
            only_node_bool = torch.logical_and(node_and_temp_edge, typ_0_bool)
            each_obj_bool[:, obj_idx, :] = only_node_bool
        
        batch = {            
            'input' : {
                'num_objs'          : b_num_objs,
                'token_pair_idx'    : p_token_pair_idx,                 # B, num_clip, max_tok_len, 2
                'token_pair_time'   : p_token_pair_time,                # B, num_clip, max_tok_len, 2
                'token_types'       : p_token_types,                    # B, num_clip, max_tok_len
                'pad_mask'          : p_pad_mask,                       # B, num_clip, max_tok_len
                'attr_feats_lookup' : p_attr_feats,                     # B, max_objs_in_batch, 200
                'coord_feats'       : p_coord_feat,                     # B, num_clip, max_tok_len, 24
                'times'             : p_token_pair_time[:, :, :, :1],   # B, num_clip, max_tok_len, 1
                'idx_in_lookup'     : p_idx_in_lookup,                  # B, num_clip, max_tok_len, 2
                'n_id_lookup'       : p_node_ids,                       # list of torch.Tensor shape [num_clip, clip_len*num_obj]
                'each_obj_bool'     : each_obj_bool,                    # B, MAX_OBJS_IN_BATCH, NC*MAX_TOKEN_LEN
                'obj_mask'          : obj_mask                          # B, MAX_OBJS_IN_BATCH
            },

            'target' : {
                'vid_action'        : p_tgt_vid_action,                                           # B, num_actions(14 or 301)
                'comp_motion'       : p_tgt_perobj_action if self.dataset_cfg.use_edm else None,  # B, MAX_OBJS_IN_BATCH, 4
                'comp_object'       : p_tgt_perobj_object if self.dataset_cfg.use_edm else None,  # B, MAX_OBJS_IN_BATCH, 5
                'comp_TR'           : p_tgt_trs           if self.dataset_cfg.use_edm else None,  # B, 4*MAX_OBJS, 4*MAX_OBJS, 3
                'edm_aa_idx'        : p_tgt_aas           if self.dataset_cfg.use_edm else None,
            },
        }
        
        return batch