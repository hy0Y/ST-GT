# from __future__ import annotations

import os
import logging
import numpy as np
import json
import torch

# from typing import List, Tuple, Dict
from omegaconf import DictConfig

from torch.nn.utils.rnn import pad_sequence
from utils.misc import *

# token types : Node, Spatial Edge, Temporal Edge
TYPE_N = 0
TYPE_SE = 1
TYPE_TE = 2

class MomaDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            mode : str, 
            dataset_cfg : DictConfig,
            **kwargs
        ):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        # MOMA -> 8 frames are sampled from the given clip -> FPC == 8
        self.dataset_cfg = dataset_cfg
        self.mode = mode

        # load prprocessed data (input)
        self.video_ids = self.get_video_ids(mode=mode)

        # load prprocessed data (target)
        self.vid2target = self.get_main_targets(self.video_ids, mode=mode)

    def get_video_ids(self, mode: str):
        logging.info(f'{mode} dataset : get video ids')

        split_pth = os.path.join(self.dataset_cfg.split_dir, f'{mode}_split.txt')
        with open(split_pth, 'r') as f: 
            vids = [line.rstrip() for line in f.readlines()]
        return vids

    def get_metadatas(self, vid):
        mtdata_pth = os.path.join(self.dataset_cfg.metadata_dir, 'basic_info', f'{vid}.npy')
        metadata = np.load(mtdata_pth, allow_pickle=True).item()
        return metadata

    def get_objidx_perclip(self, num_insts_pc):
        # TODO integrate to metadata
        result = []
        current_number = 1
        for length in num_insts_pc:
            sublist = list(range(current_number, current_number + length))
            result.append(sublist)
            current_number += length
        return result

    def get_main_targets(self, vids, mode):
        main_target = 'vid2subactivity.json'
        with open(os.path.join(self.dataset_cfg.target_dir, main_target), 'r') as j:
            targets = json.load(j)['vid2idx']
        return {vid : target for vid, target in targets.items() if vid in vids}

    def __len__(self):
        return len(self.video_ids)
        
    def __getitem__(self, idx):
        
        # get current video ID
        vid = self.video_ids[idx]

        # get metadatas
        metadata = self.get_metadatas(vid)

        num_objs_pc = metadata['num_insts_perclip']

        # token type : Spatial Edge, node idx pair
        Spatial_E_pair_idx = metadata['spatial_edge_idxs'] 

        # token type : Spatial Edge, time pair
        Spatial_E_pair_time = metadata['spatial_time_idxs']

        EIDX_in_lookup = metadata['eidxs_in_lookup']

        # token type : Node
        Node_pair_idx, Node_pair_time = [], []              # num_clip, total nodes in clip 

        # token type : Temporal Edge
        Temporal_E_pair_idx, Temporal_E_pair_time = [], []  # num_clip, (num_frame-1)*objs

        objidx_pc = self.get_objidx_perclip(num_objs_pc)
        for clip_idx, sampled_frs in enumerate(metadata['frame_info']):
            
            cur_objidxs = objidx_pc[clip_idx]

            num_frames = self.dataset_cfg.FPC

            # Token type : Node, node idx pair
            Node_pair_idx.append(
                [(obj_idx, obj_idx) for obj_idx in cur_objidxs] * num_frames
            )

            # Token type : Node, time pair
            cur_clip_node_time_idxs = []
            for ts in sampled_frs:
                cur_clip_node_time_idxs.extend([(ts, ts) for _ in cur_objidxs])
            Node_pair_time.append(cur_clip_node_time_idxs)

            # Token type : Temporal Edge, node idx pair
            Temporal_E_pair_idx.append(
                [(obj_idx, obj_idx) for obj_idx in cur_objidxs] * (num_frames - 1) 
            )

            # Token type : Temporal Edge, time pair
            cur_clip_temp_edge_time_idxs = []
            for bef_ts, aft_ts in zip(sampled_frs, sampled_frs[1:]):
                cur_clip_temp_edge_time_idxs.extend([(bef_ts, aft_ts) for _ in cur_objidxs])
            Temporal_E_pair_time.append(cur_clip_temp_edge_time_idxs)
        
        # aggregate 3 different type's tokens in one list
        NC = len(metadata['frame_info'])

        token_pair_idx, token_pair_time, token_types = [], [], []

        token_eidx_in_lookup = []

        pad_mask = []

        for clip_idx in range(NC):

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

            token_eidx_in_lookup.append(
                torch.Tensor(
                    [0] * len(Node_pair_idx[clip_idx]) + \
                    EIDX_in_lookup[clip_idx] + \
                    [0] * len(Temporal_E_pair_idx[clip_idx])
                )
            )

        # moma required..
        token_t_idx_in_adj = []
        for clip_idx, global_time_pairs in enumerate(token_pair_time):
            sampled_frs = metadata['frame_info'][clip_idx]

            token_t_idx_in_adj.append(
                torch.Tensor(
                    [[sampled_frs.index(_1_t_idx), sampled_frs.index(_2_t_idx)] for _1_t_idx, _2_t_idx in global_time_pairs]
                )
            )

        # get node, edge feature
        n_feats = torch.from_numpy(np.load(f'{self.dataset_cfg.feat_dir}/nfeat/{vid}.npy'))
        e_feats = torch.from_numpy(np.load(f'{self.dataset_cfg.feat_dir}/efeat/{vid}.npy'))

        # get bbox feature
        objbytsby4d = torch.from_numpy(np.load(f'{self.dataset_cfg.bbox_dir}/{vid}.npy'))    # #node + 1, ts, 4
        bbox_feat = []
        for idx, (clip_token_pair_idx, clip_token_pair_time) in enumerate(zip(token_pair_idx, token_pair_time)):
            # T, 2
            T = len(clip_token_pair_idx)

            bbox_feat_clip = torch.zeros((T, 8))

            _1_n_idx = clip_token_pair_idx[:, 0].long()
            _2_n_idx = clip_token_pair_idx[:, 1].long()

            _1_t_idx = clip_token_pair_time[:, 0].long()
            _2_t_idx = clip_token_pair_time[:, 1].long()

            bbox_feat_clip[:, :4] = objbytsby4d[_1_n_idx, _1_t_idx]  # current clip's tokens, 12
            bbox_feat_clip[:, 4:] = objbytsby4d[_2_n_idx, _2_t_idx]  # current clip's tokens, 12

            bbox_feat.append(bbox_feat_clip)    # T, 8

        clip_len = self.dataset_cfg.FPC
        local_max_num_objs = max(num_objs_pc)
        _minus_idx = np.cumsum([0] + num_objs_pc[:-1])
        idx_in_adj = []
        for clip_idx, (clip_token_pair_idx, clip_token_pair_time, t_idx_in_adj) in enumerate(zip(token_pair_idx, token_pair_time, token_t_idx_in_adj)):

            _minus = _minus_idx[clip_idx]
            nidx_in_clip = torch.Tensor(clip_token_pair_idx) - 1 - _minus             # 536, 2

            idx_in_adj_clip = nidx_in_clip + local_max_num_objs * t_idx_in_adj
            idx_in_adj.append(idx_in_adj_clip)

        # NOTE time complexity... TODO to GPU  #clip, 12개 클립중 obj max * FPC, 12개 클립중 obj max * FPC
        n_ids = torch.from_numpy(np.load(f'{self.dataset_cfg.metadata_dir}/random_gaussian/{vid}.npy'))
        
        token_pair_idx = pad_sequence(token_pair_idx, batch_first=True, padding_value=0)            # [12, 297, 2]
        token_pair_time = pad_sequence(token_pair_time, batch_first=True, padding_value=0)          # [12, 297, 2]
        token_types = pad_sequence(token_types, batch_first=True, padding_value=0)                  # [12, 297]

        pad_mask = pad_sequence(pad_mask, batch_first=True, padding_value=0)                        # [12, 297]

        token_eidx_in_lookup = pad_sequence(token_eidx_in_lookup, batch_first=True, padding_value=0)            

        bbox_feat = pad_sequence(bbox_feat, batch_first=True, padding_value=0)                      # [12, 297, 8]
        idx_in_adj = pad_sequence(idx_in_adj, batch_first=True, padding_value=0)                    # [12, 297, 2]

        NC_mask = torch.ones(NC)

        target = self.vid2target[vid]
        target = one_hot_multilabel(target, num_classes=self.dataset_cfg.num_class).tolist()

        data = {

            'num_objs' : sum(num_objs_pc),                  
            'tokens' :  token_pair_idx.shape[1],    # num_tokens
            'nc_mask' : NC_mask,

            'token_pair_idx'  : token_pair_idx,     #          -> per clip
            'token_pair_time' : token_pair_time,    #         -> per clip
            'token_types'     : token_types,        #         -> per clip

            'pad_mask'        : pad_mask,           #           -> per clip
            'token_eidx'      : token_eidx_in_lookup,

            'n_feats_lup'     : n_feats,            # 92, 384     -> per video
            'e_feats_lup'     : e_feats,            # 2720, 384   -> per video
            'bbox_feat'       : bbox_feat,          #     -> per clip

            'idx_in_adj'      : idx_in_adj,         #       -> per clip
            'lap_eigens'      : n_ids,              #     -> per clip

            'target'          : target,
        }
        return data

    def custom_collate_fn(self, batch):
        B = len(batch)
        # NC = self.num_clips
        MAX_TOKEN_LEN = max([x.shape[1] for x in [data['token_pair_idx'] for data in batch]])
        MAX_OBJS_IN_BATCH = max([data['num_objs'] for data in batch])

        # meta data
        b_num_objs = [data['num_objs'] for data in batch]
        b_tokens = [data['tokens'] for data in batch]
        b_NC_mask = [data['nc_mask'] for data in batch]

        # inputs
        b_token_pair_idx = [data['token_pair_idx'] for data in batch]
        b_token_pair_time = [data['token_pair_time'] for data in batch]
        b_token_types = [data['token_types'] for data in batch]

        b_pad_mask = [data['pad_mask'] for data in batch]
        b_token_eidx = [data['token_eidx'] for data in batch]

        b_n_feats_lup = [data['n_feats_lup'] for data in batch]
        b_e_feats_lup = [data['e_feats_lup'] for data in batch]
        b_bbox_feat = [data['bbox_feat'] for data in batch]

        b_idx_in_adj = [data['idx_in_adj'] for data in batch]
        b_lap_eigens = [data['lap_eigens'] for data in batch]

        # target
        b_target = [data['target'] for data in batch]

        b_NC = [len(_t) for _t in b_NC_mask]
        MAX_NC = max(b_NC)
        MAX_objsxFPC = max([x.shape[1] for x in b_lap_eigens])

        # pad to token dimension
        p_token_pair_idx = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN, 2)
        p_token_pair_time = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN, 2)

        p_token_eidx = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN)
        p_token_types = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN)

        p_pad_mask = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN)
        p_bbox_feat = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN, 8)
        p_idx_in_adj = torch.zeros(B, MAX_NC, MAX_TOKEN_LEN, 2)
        p_lap_eigens = torch.zeros(B, MAX_NC, MAX_objsxFPC, MAX_objsxFPC) # NOTE temporally 

        for batch_idx in range(B):
            cur_len = b_tokens[batch_idx]
            cur_NC = b_NC[batch_idx]
            rows = b_lap_eigens[batch_idx].shape[1]

            p_token_pair_idx[batch_idx, :cur_NC, :cur_len, :] = b_token_pair_idx[batch_idx]
            p_token_pair_time[batch_idx, :cur_NC, :cur_len, :] = b_token_pair_time[batch_idx]

            p_token_eidx[batch_idx, :cur_NC, :cur_len] = b_token_eidx[batch_idx]
            p_token_types[batch_idx, :cur_NC, :cur_len] = b_token_types[batch_idx]

            p_pad_mask[batch_idx, :cur_NC, :cur_len] = b_pad_mask[batch_idx]
            p_bbox_feat[batch_idx, :cur_NC, :cur_len, :] = b_bbox_feat[batch_idx]
            p_idx_in_adj[batch_idx, :cur_NC, :cur_len, :] = b_idx_in_adj[batch_idx]
            p_lap_eigens[batch_idx, :cur_NC, :rows, :rows] = b_lap_eigens[batch_idx]

        p_n_feats = pad_sequence(b_n_feats_lup, batch_first=True, padding_value=0)
        p_e_feats = pad_sequence(b_e_feats_lup, batch_first=True, padding_value=0)
        
        p_NC_mask = pad_sequence(b_NC_mask, batch_first=True, padding_value=0)

        batch = {

            'input' : {
                'num_objs'          : b_num_objs,
                'nc_mask'           : p_NC_mask,
                'token_pair_idx'    : p_token_pair_idx,                 # B, num_clip, max_tok_len, 2
                'token_pair_time'   : p_token_pair_time,                # B, num_clip, max_tok_len, 2
                'token_types'       : p_token_types,                    # B, num_clip, max_tok_len
                'pad_mask'          : p_pad_mask,                       # B, num_clip, max_tok_len

                'token_eidx'        : p_token_eidx,                       # B, num_clip, max_tok_len

                'nfeats_lup'        : p_n_feats,                        # B, max_objs_in_batch, 200
                'efeats_lup'        : p_e_feats,                        # B, max_objs_in_batch, 200
                'bbox_feats'        : p_bbox_feat,
                'idx_in_adj'        : p_idx_in_adj,                     # B, num_clip, max_tok_len, 2
                'n_id_lookup'       : p_lap_eigens,                     # list of torch.Tensor shape [num_clip, clip_len*num_obj]
            },

            'target' : {
                'vid_action'        : torch.Tensor(b_target)
            },

        }

        return batch