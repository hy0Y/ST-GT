# from __future__ import divsision

import os
import random

import numpy as np
import torch

from itertools import accumulate
from itertools import combinations as comb

from typing import Optional

def edge_by_distance(*locs, num_nodes, num_edges, dist):
    '''
        Args :
            *locs 
                : all elements in current rows, 3 * num_nodes
                EX)
                    column name -> obj1_x, obj1_y, obj1_z , obj2_x, obj2_y, obj2_z, ... -> # of obj(node) * 3
            dist 
                : if lower than this dist, we form edge of two nodes
        
        Description :
            get each node's coordinate -> compute all pair distance and return if there is edge
            num_nodes, 3 -> num_nodes_C_2(possible edges given num_nodes), 1
    '''
    locs = np.array(list(locs)).reshape(num_nodes, 3)
    node_pair_idx = list(map(list, comb([i for i in range(num_nodes)], 2))) # num_node C 2
    edges = -1 * np.ones(num_edges)
    for idx, (src, dec) in enumerate(node_pair_idx):    # num_edge C 2
        edges[idx] = (np.linalg.norm(locs[src, :] - locs[dec, :]) < dist)
    assert -1 not in list(edges)
    return list(edges)

def rbf(D, num_rbf):
    '''
        From https://github.com/jingraham/neurips19-graph-protein-design
    '''
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    # D_expand = np.unsqueeze(D, -1)
    D_expand = np.expand_dims(D, axis=-1)
    RBF = np.exp(-((D_expand - D_mu) / D_sigma)**2)

    return RBF

def positional_embeddings(d, num_embeddings=16, period_range=[2, 1000]):
    '''
        From https://github.com/jingraham/neurips19-graph-protein-design
    '''
    frequency = np.exp(np.float32(np.arange(0, num_embeddings, 2)) * -(np.log(10000.0) / num_embeddings))
    angles = d * frequency
    E = np.concatenate((np.cos(angles), np.sin(angles)), -1)
    return E

def to_edge_feats(*locs, node_ts_coor, rbf_dim, pos_dim, total_dim):

    locs = list(locs)
    src_idx = locs[0] - 1
    dsc_idx = locs[1] - 1
    ts = int(locs[2])

    src_coor = node_ts_coor[src_idx, ts]
    dsc_coor = node_ts_coor[dsc_idx, ts]

    delta = src_coor - dsc_coor
    delta_norm = np.linalg.norm(delta)

    unit_delta = delta / delta_norm

    pos_feat = positional_embeddings(delta_norm, num_embeddings=pos_dim)

    rbf_feat = rbf(delta_norm, num_rbf=rbf_dim).squeeze()

    edge_feats = np.concatenate((rbf_feat, pos_feat, unit_delta), -1).tolist()
    
    assert len(edge_feats) == total_dim
    
    return edge_feats

def np_pad_sequence(list_of_insts, padding_value):
    batch_size = len(list_of_insts)
    cur_len = [inst.shape[0] for inst in list_of_insts]
    max_len = max([inst.shape[0] for inst in list_of_insts]) # ex 180
    np_type = 'object' if list_of_insts[0].dtype == '<U16' else list_of_insts[0].dtype
    padded = np.zeros((batch_size, max_len)).astype(np_type)
    for batch_idx, inst in enumerate(list_of_insts):
        padded[batch_idx, :] = np.pad(inst, (0, max_len - inst.shape[0]), 'constant', constant_values=padding_value)
    return padded, max_len

def multiply_accumulate(num_neighbors):
    cum_mul, split_idx = 1, []
    for ngh in num_neighbors:
        cum_mul *= ngh
        split_idx.append(cum_mul)
    return [0] + list(accumulate(split_idx))

def new_axis_with_arange(np_l):
    '''add element index to new dimension'''
    assert len(np_l.shape) == 1
    return np.hstack([np_l.reshape(-1, 1), np.arange(len(np_l)).reshape(-1, 1)])

def flatten_all(list_of_ndarray):
    return [arr.reshape(-1) if isinstance(arr, np.ndarray) else arr for arr in list_of_ndarray[0]]

def one_hot_multilabel(class_indices, num_classes=None):
    one_hot = np.zeros(num_classes)
    one_hot[class_indices] = 1
    return one_hot

def sigmoid(x): return 1 / (1 + np.exp(-x))

def adj_list(df, num_nodes):
    src_l = df.u.values
    dst_l = df.i.values
    e_idx_l = df.idx.values
    ts_l = df.ts.values

    max_idx = num_nodes
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    return adj_list

def remove_column(input, remove_idx):
    '''
        input : torch.Tensor of shape B, C or B, T, C
    '''
    orig_input_len = len(input.shape)
    rm_dim = len(remove_idx)
    if orig_input_len == 3:
        B, T, C = input.shape
        input = input.view(B*T, C)
    elif orig_input_len == 2:
        B, C = input.shape

    mask = torch.ones(C, dtype=torch.bool)
    mask[remove_idx] = False

    input = input[:, mask]

    if orig_input_len == 3:
        input = input.view(B, T, C-rm_dim)
    
    return input

def return_none(*args, **kargs):
    return None

def sum_all(*items):
    return sum(items)

def get_wandb(project_name: str, run_name: str, model):
    import wandb
    wandb.init(project=project_name)
    wandb.run.name = run_name
    wandb.run.save()
    # wandb.config.update(cfg)
    wandb.watch(model)
    return wandb



def compute_average_precision(groundtruth, predictions, false_negatives=0):
    """
    Computes average precision for a binary problem. This is based off of the
    PASCAL VOC implementation.

    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.
        false_negatives (int or None): In some tasks, such as object
            detection, not all groundtruth will have a corresponding prediction
            (i.e., it is not retrieved at _any_ score threshold). For these
            cases, use false_negatives to indicate the number of groundtruth
            instances that were not retrieved.

    Returns:
        Average precision.

    """
    predictions = np.asarray(predictions).squeeze()
    groundtruth = np.asarray(groundtruth, dtype=float).squeeze()

    if predictions.ndim == 0:
        predictions = predictions.reshape(-1)

    if groundtruth.ndim == 0:
        groundtruth = groundtruth.reshape(-1)

    if predictions.ndim != 1:
        raise ValueError(f'Predictions vector should be 1 dimensional, not '
                         f'{predictions.ndim}. (For multiple labels, use '
                         f'`compute_multiple_aps`.)')
    if groundtruth.ndim != 1:
        raise ValueError(f'Groundtruth vector should be 1 dimensional, not '
                         f'{groundtruth.ndim}. (For multiple labels, use '
                         f'`compute_multiple_aps`.)')

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1] + false_negatives

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recalls = tp / num_positives

    # Append end points of the precision recall curve.
    precisions = np.concatenate(([0.], precisions))
    recalls = np.concatenate(([0.], recalls))

    # Find points where prediction score changes.
    prediction_changes = set(
        np.where(predictions[1:] != predictions[:-1])[0] + 1)

    num_examples = predictions.shape[0]

    # Recall and scores always "change" at the first and last prediction.
    c = prediction_changes | set([0, num_examples])
    c = np.array(sorted(list(c)), dtype=np.int64)

    precisions = precisions[c[1:]]

    # Set precisions[i] = max(precisions[j] for j >= i)
    # This is because (for j > i), recall[j] >= recall[i], so we can always use
    # a lower threshold to get the higher recall and higher precision at j.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    ap = np.sum((recalls[c[1:]] - recalls[c[:-1]]) * precisions)

    return ap


def compute_multiple_aps(groundtruth, predictions, false_negatives=None):
    """Convenience function to compute APs for multiple labels.

    Args:
        groundtruth (np.array): Shape (num_samples, num_labels)
        predictions (np.array): Shape (num_samples, num_labels)
        false_negatives (list or None): In some tasks, such as object
            detection, not all groundtruth will have a corresponding prediction
            (i.e., it is not retrieved at _any_ score threshold). For these
            cases, use false_negatives to indicate the number of groundtruth
            instances which were not retrieved for each category.

    Returns:
        aps_per_label (np.array, shape (num_labels,)): Contains APs for each
            label. NOTE: If a label does not have positive samples in the
            groundtruth, the AP is set to -1.
    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth)
    if predictions.ndim != 2:
        raise ValueError('Predictions should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))
    if groundtruth.ndim != 2:
        raise ValueError('Groundtruth should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))

    num_labels = groundtruth.shape[1]
    aps = np.zeros(groundtruth.shape[1])
    if false_negatives is None:
        false_negatives = [0] * num_labels
    for i in range(num_labels):
        if not groundtruth[:, i].any():
            print('WARNING: No groundtruth for label: %s' % i)
            aps[i] = -1
        else:
            aps[i] = compute_average_precision(groundtruth[:, i],
                                               predictions[:, i],
                                               false_negatives[i])
    return aps


def lap_eig(dense_adj, number_of_nodes, in_degree):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    dense_adj = dense_adj.detach().cpu().float().numpy()
    in_degree = in_degree.detach().cpu().float().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - N @ A @ N

    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(L)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def get_pe_2d(
        dense_adj,
        edge12_indices: torch.LongTensor, 
        n_node: int, 
        n_edge12: int,
        half_pos_enc_dim=128
    ):
    '''
        ~ 로부터 채택 및 수정
    '''
    assert n_node <= half_pos_enc_dim
    device = edge12_indices.device
    
    in_degree = dense_adj.long().sum(dim=1).view(-1)

    EigVec, EigVal = lap_eig(dense_adj, n_node, in_degree)  # EigVec: [N, N] 
    node_pe = torch.zeros(n_node, half_pos_enc_dim).to(device)  # [N, half_pos_enc_dim]
    node_pe[:, :n_node] = EigVec
    E = edge12_indices.shape[0]
    all_edges_pe = torch.zeros([E, 2 * half_pos_enc_dim]).to(device)
    all_edges_pe[:n_edge12, :half_pos_enc_dim] = torch.index_select(node_pe, 0, edge12_indices[:n_edge12, 0])
    all_edges_pe[:n_edge12, half_pos_enc_dim:] = torch.index_select(node_pe, 0, edge12_indices[:n_edge12, 1])

    return all_edges_pe.unsqueeze(0)  # [1, E, 2*half_pos_enc_dim]


# # NOTE preprocess infos
# self.make_clipinfos_laplacian(mode=mode, num_clips=self.num_clips, edge_dist=self.edge_dist)


# # cater preprocess
    # def make_clipinfos_laplacian(self, mode, num_clips, edge_dist):
    #     assert mode in ['train', 'val', 'test']
    #     logging.info(f'{mode} dataset : get clipinfos')

    #     graph_dir = self.dataset_cfg.path.graph_dir
    #     task = self.task
    #     graph_csv_name = f'{mode}_dist2edge_{edge_dist}.csv'
    #     graph_pth = os.path.join(graph_dir, task, mode, graph_csv_name)
    #     graph_df = pd.read_csv(graph_pth)

    #     # vid : [[s, e], [s, e], [s, e], [s, e], ... , [s, e]]
    #     vid2clipinfos = dict()
    #     for vid in tqdm(self.video_names):
    #         start = 0                       # CG땐 이 세부분을 각 clip 정보로
    #         end = 301
    #         N_FRAMES = end - start

    #         clip_len = N_FRAMES // num_clips

    #         # current video's  Spatiotemporal graph
    #         graph = graph_df[graph_df['vid'] == vid]

    #         # current video's num objects
    #         num_objs = len(self.vid2mapping[vid])
            
    #         frame_info = []
    #         spatial_edge_idxs, spatial_time_idxs = [], []
    #         adj_matrix = np.zeros((10, num_objs*30, num_objs*30))
    #         for clip_idx in range(num_clips):
    #             # frame info
    #             s = start + clip_idx * clip_len
    #             e = start + (clip_idx + 1) * clip_len
    #             frame_info.append([s, e])

    #             # spatial edge infos
    #             cur_clip_df = graph[(graph.ts >= s) & (graph.ts < e)]

    #             if len(cur_clip_df) == 0:
    #                 spatial_edge_idxs.append([])
    #                 spatial_time_idxs.append([])
    #             else:
    #                 srcs = cur_clip_df.u.values
    #                 dess = cur_clip_df.i.values
    #                 tss = cur_clip_df.ts.values

    #                 cur_clip_sp_edge_idxs, cur_clip_sp_time_idxs = [], []
    #                 for src, des, tss in zip(srcs, dess, tss):
    #                     cur_clip_sp_edge_idxs.append((src, des))
    #                     cur_clip_sp_time_idxs.append((int(tss), int(tss)))

    #                 spatial_edge_idxs.append(cur_clip_sp_edge_idxs)
    #                 spatial_time_idxs.append(cur_clip_sp_time_idxs)
                
    #             clip_adj = []
    #             for ts in range(s, e):

    #                 clip_ts = ts % (e - s)
                    
    #                 for obj_idx in range(1, num_objs + 1):
                        
    #                     # temporal edge
    #                     if ts != e - 1:
    #                         clip_adj.append(
    #                             [(obj_idx - 1) + (num_objs * clip_ts), (obj_idx - 1) + (num_objs * (clip_ts + 1))]
    #                         )
    #                         clip_adj.append(
    #                             [(obj_idx - 1) + (num_objs * (clip_ts + 1)), (obj_idx - 1) + (num_objs * clip_ts)]
    #                         )

    #                     # spatial edge
    #                     if len(cur_clip_df[cur_clip_df.ts == ts]) != 0:
    #                         cur_time_df = cur_clip_df[cur_clip_df.ts == ts]
    #                         srcs = cur_time_df.u.values - 1
    #                         dess = cur_time_df.i.values - 1

    #                         srcs = srcs + (num_objs * clip_ts)
    #                         dess = dess + (num_objs * clip_ts)

    #                         for src, des in zip(srcs, dess):
    #                             clip_adj.append(
    #                                 [src, des]
    #                             )
    #                             clip_adj.append(
    #                                 [des, src]
    #                             )
                
    #             clip_adj = np.array(clip_adj)
    #             adj_matrix[clip_idx, clip_adj[:, 0], clip_adj[:, 1]] = 1

    #         np.save(f'/data/max2action/preprocessed/laplacian/ED_1.5_clips_10/{vid}.npy', adj_matrix)
                


    #         vid2clipinfos.update({
    #             vid : {'frame_info' :  frame_info,
    #                    'spatial_edge_idxs' : spatial_edge_idxs,
    #                    'spatial_time_idxs' : spatial_time_idxs,
    #                 #    'spatio_temporal_adjs' : spatio_temporal_adjs
    #             }
    #         })
    #         clipinfo = {
    #             'frame_info' :  frame_info,
    #             'spatial_edge_idxs' : spatial_edge_idxs,
    #             'spatial_time_idxs' : spatial_time_idxs
    #         }
            

    #         np.save(f'/data/max2action/preprocessed/tokenGT/clipinfos/ED_1.5_NC_10/{vid}.npy', clipinfo)
        
    #     print('good')
    #     exit()


# NOTE preprocess infos
# self.make_clipinfos_laplacian(mode=mode, FPC=self.FPC) # NOTE 일단 8


# # moma preprocess
    # def make_clipinfos_laplacian(self, mode, FPC):
    #     assert mode in ['train', 'val', 'test']

    #     vid2clipinfos = dict()
    #     for vid in tqdm(self.video_names):
            
    #         graph_pth = os.path.join(self.dataset_cfg.path.graph_dir, vid)

    #         graph_df = pd.read_csv(f'{graph_pth}.csv')

    #         num_objs = sum([len(key2idx) for sact_id, key2idx in self.vid2mapping[vid].items()])

    #         NS = len(graph_df['subclip_id'].unique())
            
    #         frame_info = []
    #         spatial_edge_idxs, spatial_time_idxs = [], []
    #         edge_idx_in_lookup = []

    #         adj_matrix = np.zeros((NS, num_objs*FPC, num_objs*FPC))   # NS, num_objs*FPC, num_objs*FPC

    #         for clip_idx, (_, clip_df) in enumerate(graph_df.groupby('subclip_id')):
                               
    #             s, e = int(math.floor(clip_df.iloc[0].ts)), int(math.ceil(clip_df.iloc[-1].ts))

    #             sampled_frames = list(np.linspace(s, e-1, num=FPC).astype(int))

    #             frame_info.append(sampled_frames)

    #             cur_clip_sp_edge_idxs, cur_clip_sp_time_idxs = [], []

    #             cur_clip_edge_idx_in_lookup = []

    #             for fr in sampled_frames:
    #                 srcs = clip_df[(clip_df.ts >= fr) & (clip_df.ts < fr+1)].u.values
    #                 dess = clip_df[(clip_df.ts >= fr) & (clip_df.ts < fr+1)].i.values
    #                 tss = clip_df[(clip_df.ts >= fr) & (clip_df.ts < fr+1)].ts.values
    #                 eidxs = clip_df[(clip_df.ts >= fr) & (clip_df.ts < fr+1)].idx.values

    #                 for src, des, tss, eidx in zip(srcs, dess, tss, eidxs):
    #                     cur_clip_sp_edge_idxs.append((src, des))
    #                     cur_clip_sp_time_idxs.append((int(tss), int(tss)))

    #                     cur_clip_edge_idx_in_lookup.append(eidx)
                
    #             spatial_edge_idxs.append(cur_clip_sp_edge_idxs)
    #             spatial_time_idxs.append(cur_clip_sp_time_idxs)
    #             edge_idx_in_lookup.append(cur_clip_edge_idx_in_lookup)

    #             # clip_adj = []
    #             # for fr_idx, global_ts in enumerate(sampled_frames):

    #             #     # clip_ts = ts % (e - s) - s
    #             #     # clip_ts = global_ts - s
                    
    #             #     for obj_idx in range(1, num_objs + 1):
                        
    #             #         # temporal edge
    #             #         if global_ts != e - 1:
    #             #             clip_adj.append(
    #             #                 [(obj_idx - 1) + (num_objs * fr_idx), (obj_idx - 1) + (num_objs * (fr_idx + 1))]
    #             #             )
    #             #             clip_adj.append(
    #             #                 [(obj_idx - 1) + (num_objs * (fr_idx + 1)), (obj_idx - 1) + (num_objs * fr_idx)]
    #             #             )
                        
    #             #         # spatial edge
    #             #         if len(clip_df[(clip_df.ts >= global_ts) & (clip_df.ts < global_ts+1)]) != 0:
    #             #             cur_time_df = clip_df[(clip_df.ts >= global_ts) & (clip_df.ts < global_ts+1)]
    #             #             srcs = cur_time_df.u.values - 1
    #             #             dess = cur_time_df.i.values - 1

    #             #             srcs = srcs + (num_objs * fr_idx)
    #             #             dess = dess + (num_objs * fr_idx)

    #             #             for src, des in zip(srcs, dess):
    #             #                 clip_adj.append(
    #             #                     [src, des]
    #             #                 )
    #             #                 clip_adj.append(
    #             #                     [des, src]
    #             #                 )

    #             # clip_adj = np.array(clip_adj)
    #             # adj_matrix[clip_idx, clip_adj[:, 0], clip_adj[:, 1]] = 1

    #         # np.save(f'/data/MOMA/preprocessed/clipinfos/adjmatrix/{vid}.npy', adj_matrix)

    #         # clipinfo = {
    #         #     'frame_info' :  frame_info,
    #         #     'spatial_edge_idxs' : spatial_edge_idxs,
    #         #     'spatial_time_idxs' : spatial_time_idxs,
    #         #     'eidxs_in_lookup' : edge_idx_in_lookup
    #         # }
            
    #         # np.save(f'/data/MOMA/preprocessed/clipinfos/newinfo/{vid}.npy', clipinfo)


