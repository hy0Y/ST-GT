import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import permutations, product

class MLP(nn.Module):
    def __init__(self, in_features, out_features, nlayers, **kwargs):
        super().__init__()
        layers = [[nn.Linear(in_features, in_features),
                   nn.ReLU()] for _ in range(nlayers - 1)]
        # flatten out the pairs
        layers = [item for sublist in layers for item in sublist]
        layers.append(nn.Linear(in_features, out_features))
        self.cls = nn.Sequential(*layers)

    def forward(self, inp):
        return self.cls(inp)

class MeanPoolMLP(nn.Module):
    def __init__(self, in_features, out_features, nlayers, **kwargs):
        super().__init__()
        self.cls = MLP(
            in_features=in_features,
            out_features=out_features,
            nlayers=nlayers
        )

    def forward(self, inp, objmask):
        '''
            inp shape : B, MAX_OBJS_IN_BATCH, dim
            out shape : 
        '''
        logits = dict()

        B, MAX_OBJS_IN_BATCH, dim = inp.shape

        # NOTE mean pooling version
        inp = inp * objmask.view(B, MAX_OBJS_IN_BATCH, 1)
        pooled_inp = inp.sum(dim=1).view(B, dim) / objmask.sum(dim=1).view(B, 1)              # B, dim
        
        final_logit = self.cls(pooled_inp)                       # B, 14
                
        logits.update({
            'final_outputs' : final_logit,
        })

        return logits
    
class EDM_task1(nn.Module):
    def __init__(
            self,
            input_dim,
            num_object,
            num_action,
            act_threshold,
            comp_map_pth,
            num_classes,
            **kwargs
        ):
        super().__init__()

        # required
        self.num_object = num_object
        self.num_action = num_action
        self.act_thres = act_threshold
        self.num_classes = num_classes

        # classifiers
        self.object_cls = nn.Linear(input_dim, num_object)
        self.action_cls = nn.Linear(input_dim, num_action)

        # composite mappings
        comp_map = np.load(comp_map_pth, allow_pickle=True).item()
        self.obj_act_to_aaidx_lookup = torch.from_numpy(comp_map['obj_act_to_aaidx_lookup'][1:, 1:])       # 5, 4

        self.not_exist_cls = nn.Linear(input_dim, num_classes)

    def forward(
            self, 
            inp, 
            objmask, 
            AAidxs_tgts,
            is_eval=False
        ):
        logits = dict()
        eps = -1e4
        NUM_ACT = self.num_action   # 4

        device = inp.device
        B, MAX_OBJS, D = inp.shape

        exp_objmasks = objmask.view(B, MAX_OBJS, 1).expand(B, MAX_OBJS, NUM_ACT).reshape(B, MAX_OBJS*NUM_ACT)

        # non exist composite action prediction
        masked_inp = inp * objmask.view(B, MAX_OBJS, 1)
        pooled_inp = masked_inp.sum(dim=1).view(B, D) / objmask.sum(dim=1).view(B, 1)              # B, dim
        non_exist_out = self.not_exist_cls(pooled_inp)

        obj_out = self.object_cls(inp)        # B, MAX_OBJS, 5
        act_out = self.action_cls(inp)        # B, MAX_OBJS, 4
        
        pred_obj_idxs = obj_out.max(dim=-1)[1]                                                                                          # B, MAX_OBJS
        pred_obj_idxs = pred_obj_idxs.view(B, MAX_OBJS, 1).expand(B, MAX_OBJS, NUM_ACT).reshape(B, MAX_OBJS*NUM_ACT)                    # B, MAX_OBJS*4

        pred_act_bools = (torch.sigmoid(act_out.view(B, MAX_OBJS*self.num_action)) > self.act_thres).to(torch.int64).cpu()              # B, MAX_OBJS*4
        pred_act_bools = pred_act_bools * exp_objmasks.cpu()                                                                            # B, MAX_OBJS*4

        if is_eval:
            # evaluation time -> use pred obj, action -> make pred aaidxs
            pad_objact_LU = torch.cat([self.obj_act_to_aaidx_lookup, torch.ones(self.num_object, 1) * -1], dim=-1).to(device)
            pred_aaidxs = torch.zeros(B, MAX_OBJS*NUM_ACT).to(device)
            selected_masks = torch.zeros(B, MAX_OBJS*NUM_ACT).to(device).bool()
            for batch_idx in range(B):
                exp_objmask = exp_objmasks[batch_idx].bool()
                pred_obj_idx = pred_obj_idxs[batch_idx]
                pred_act_bool = pred_act_bools[batch_idx].bool()    # MAX_OBJS*4

                pred_act_idx = torch.arange(self.num_action).unsqueeze(0).expand(MAX_OBJS, self.num_action).flatten().to(device)

                pred_act_idx[~pred_act_bool] = -1
                pred_aaidx = pad_objact_LU[pred_obj_idx, pred_act_idx]

                selected_mask = torch.logical_and((pred_aaidx != -1), exp_objmask)

                pred_aaidxs[batch_idx] = pred_aaidx
                selected_masks[batch_idx] = selected_mask

        else:
            # training time -> use target AAidxs,  pred_act_bools -> make pred aaidxs
            selected_masks = torch.logical_and((pred_act_bools == 1), (AAidxs_tgts != -1))      # B, MAX_OBJS*4
            pred_aaidxs = torch.empty_like(AAidxs_tgts).copy_(AAidxs_tgts)                      # copy
            pred_aaidxs[~selected_masks] = -1                                                   # B, MAX_OBJS*4

        act_temp = act_out.view(B, MAX_OBJS*self.num_action)
        pred_aa_logits = torch.empty_like(act_temp).copy_(act_temp)     # NOTE important
        pred_aa_logits[~selected_masks] = eps

        aa_outputs = torch.zeros(B, self.num_classes).to(device)
        exist_bools = torch.zeros(B, self.num_classes).to(torch.bool)
        for batch_idx in range(B):
            pred_aa_logit = pred_aa_logits[batch_idx]
            pred_aaidx = pred_aaidxs[batch_idx]
            
            # to prevent negative class error in F.one_hot below
            is_minus = (pred_aaidx == -1)
            is_minus_idx = (pred_aaidx == -1).nonzero(as_tuple=False).view(-1)
            temp_idx = 0
            pred_aaidx[is_minus] = temp_idx
            
            N = MAX_OBJS*self.num_action
            diag_logit = torch.zeros(N, N).to(device)
            diag_logit[torch.arange(N), torch.arange(N)] = pred_aa_logit

            one_hot_aaidx = F.one_hot(pred_aaidx.long(), self.num_classes).to(device)
            one_hot_aaidx[is_minus_idx, temp_idx] = 0

            mapped_logit = torch.matmul(diag_logit, one_hot_aaidx.float())
            mapped_logit = mapped_logit.max(dim=0)[0]

            aa_outputs[batch_idx] = mapped_logit
            exist_bools[batch_idx] = (mapped_logit != 0.)
        
        # combine tr outputs(exist) + non exist outputs
        final_outputs = torch.zeros(B, self.num_classes).to(device)
        final_outputs[exist_bools] = aa_outputs[exist_bools]
        final_outputs[~exist_bools] = non_exist_out[~exist_bools]
                
        logits.update({
            'final_outputs' : final_outputs,
            'perobj_action' : act_out,
            'perobj_object' : obj_out,
        })

        return logits

class EDM_task2(nn.Module):
    def __init__(
            self,
            input_dim,
            num_object,
            num_action,
            num_temporal_relations,
            act_threshold,
            embed_dim,
            comp_map_pth,
            num_classes,
            **kwargs
        ):
        super().__init__()

        # required
        self.num_object = num_object
        self.num_action = num_action
        self.num_temporal_relations = num_temporal_relations
        self.act_thres = act_threshold
        self.final_dim = 2*(input_dim+2*embed_dim)
        self.num_classes = num_classes

        # classifiers
        self.object_cls = nn.Linear(input_dim, num_object)
        self.action_cls = nn.Linear(input_dim, num_action)
        self.temporal_relation_cls = nn.Linear(self.final_dim, num_temporal_relations)

        # composite lookups
        self.object_lookup = nn.Embedding(num_embeddings=num_object, embedding_dim=embed_dim)
        self.action_lookup = nn.Embedding(num_embeddings=num_action, embedding_dim=embed_dim)

        # composite mappings
        comp_map = np.load(comp_map_pth, allow_pickle=True).item()
        self.obj_act_to_aaidx_lookup = torch.from_numpy(comp_map['obj_act_to_aaidx_lookup'][1:, 1:])       # 5, 4
        self.aa_aa_tr_to_caidx_lookup = torch.from_numpy(comp_map['aa_aa_tr_to_caidx_lookup'])             # 14+1, 14+1, 3

        self.not_exist_cls = nn.Linear(input_dim, num_classes)
    
    def forward(
            self, 
            inp, 
            objmask, 
            AAidxs_tgts,
            TR_tgts,
            is_eval=False
        ):
        logits = dict()
        NUM_ACT = self.num_action   # 4
        NUM_TRs = self.num_temporal_relations   # 3

        device = inp.device
        B, MAX_OBJS, D = inp.shape

        exp_objmasks = objmask.view(B, MAX_OBJS, 1).expand(B, MAX_OBJS, NUM_ACT).reshape(B, MAX_OBJS*NUM_ACT)

        # non exist composite action prediction
        masked_inp = inp * objmask.view(B, MAX_OBJS, 1)
        pooled_inp = masked_inp.sum(dim=1).view(B, D) / objmask.sum(dim=1).view(B, 1)              # B, dim
        non_exist_out = self.not_exist_cls(pooled_inp)

        obj_out = self.object_cls(inp)        # B, MAX_OBJS, 5
        act_out = self.action_cls(inp)        # B, MAX_OBJS, 4
        
        posb_action_idxs = torch.arange(NUM_ACT).view(1, NUM_ACT).repeat(B, MAX_OBJS).to(device)                                            # B, MAX_OBJS*4

        pred_obj_idxs = obj_out.max(dim=-1)[1]                                                                                          # B, MAX_OBJS
        pred_obj_idxs = pred_obj_idxs.view(B, MAX_OBJS, 1).expand(B, MAX_OBJS, NUM_ACT).reshape(B, MAX_OBJS*NUM_ACT)                      # B, MAX_OBJS*4

        posb_inputs = inp.view(B, MAX_OBJS, 1, D).expand(B, MAX_OBJS, NUM_ACT, D).reshape(B, MAX_OBJS*NUM_ACT, D)                         # B, MAX_OBJS*4, D
        pred_act_bools = (torch.sigmoid(act_out.view(B, MAX_OBJS*self.num_action)) > self.act_thres).to(torch.int64).cpu()       # B, MAX_OBJS*4
        pred_act_bools = pred_act_bools * exp_objmasks.cpu()        # B, MAX_OBJS*4

        if is_eval:
            # evaluation time -> use pred obj, action -> make pred aaidxs
            pad_objact_LU = torch.cat([self.obj_act_to_aaidx_lookup, torch.ones(self.num_object, 1) * -1], dim=-1).to(device)
            pred_aaidxs = torch.zeros(B, MAX_OBJS*NUM_ACT).to(device)
            selected_masks = torch.zeros(B, MAX_OBJS*NUM_ACT).to(device).bool()
            for batch_idx in range(B):
                exp_objmask = exp_objmasks[batch_idx].bool()
                pred_obj_idx = pred_obj_idxs[batch_idx]
                pred_act_bool = pred_act_bools[batch_idx].bool()    # MAX_OBJS*4

                pred_act_idx = torch.arange(self.num_action).unsqueeze(0).expand(MAX_OBJS, self.num_action).flatten().to(device)

                pred_act_idx[~pred_act_bool] = -1
                pred_aaidx = pad_objact_LU[pred_obj_idx, pred_act_idx]

                selected_mask = torch.logical_and((pred_aaidx != -1), exp_objmask)

                pred_aaidxs[batch_idx] = pred_aaidx
                selected_masks[batch_idx] = selected_mask

        else:
            # training time -> use target AAidxs,  pred_act_bools -> make pred aaidxs
            selected_masks = torch.logical_and((pred_act_bools == 1), (AAidxs_tgts != -1))      # both on cpu                  # B, MAX_OBJS*4
            pred_aaidxs = torch.empty_like(AAidxs_tgts).copy_(AAidxs_tgts)                      # copy
            pred_aaidxs[~selected_masks] = -1                                                   # B, MAX_OBJS*4

        # projection to each subspace
        posb_obj_emb = self.object_lookup(pred_obj_idxs)                                 
        posb_act_emb = self.action_lookup(posb_action_idxs)
        posb_comp_embs = torch.cat([posb_inputs, posb_obj_emb, posb_act_emb], dim=-1)

        num_pairs = []
        for batch_idx in range(B):
            selected_mask = selected_masks[batch_idx]
            idx_in_AA = (selected_mask == True).nonzero(as_tuple=False).view(-1).tolist()
            num_pairs.append(len(list(permutations(idx_in_AA, 2))))
            # num_pairs.append(len(list(product(idx_in_AA, repeat=2))))
        MAX_NUM_PAIRS = max(num_pairs)
        
        pair_outputs = (torch.ones(B, MAX_NUM_PAIRS, self.final_dim) * -1).to(device)
        TR_targets = (torch.ones(B, MAX_NUM_PAIRS, NUM_TRs) * -1).to(device)
        pair_aaidxs = (torch.ones(B, MAX_NUM_PAIRS, 2) * -1).to(device)
        caidxs = (torch.ones(B, MAX_NUM_PAIRS, NUM_TRs) * -1).to(device)

        for batch_idx in range(B):
            # retrieve current batch infos
            posb_comp_emb = posb_comp_embs[batch_idx]
            selected_mask = selected_masks[batch_idx]
            pred_aaidx = pred_aaidxs[batch_idx]
            TR_tgt = TR_tgts[batch_idx]

            # get matched idx_in_AAlist
            idx_in_AA = (selected_mask == True).nonzero(as_tuple=False).view(-1).tolist()
            pred_aaidx = pred_aaidx[pred_aaidx != -1]

            assert len(idx_in_AA) == len(pred_aaidx)

            # permutate idxs and make pair
            pair_idx = torch.Tensor(list(permutations(idx_in_AA, 2))).long()                          # 12, 2
            pair_aaidx = torch.Tensor(list(permutations(pred_aaidx, 2))).long()                       # 12, 2

            if len(pair_idx) == 0:
                continue
            
            pair_aaidxs[batch_idx, :len(pair_idx), :] = pair_aaidx

            caidxs[batch_idx, :len(pair_idx), :] = self.aa_aa_tr_to_caidx_lookup[pair_aaidx[:, 0], pair_aaidx[:, 1]]

            # target
            TR_targets[batch_idx, :len(pair_idx), :] = TR_tgt[pair_idx[:, 0], pair_idx[:, 1]]

            # pair embed
            pair_outputs[batch_idx, :len(pair_idx), :] = torch.cat(
                [posb_comp_emb[pair_idx[:, 0]], posb_comp_emb[pair_idx[:, 1]]], dim=-1
            )
        
        pair_logits = self.temporal_relation_cls(pair_outputs)                                       # B, max_num_pairs, 3
        
        tr_outputs = torch.zeros(B, self.num_classes).to(device)
        exist_bools = torch.zeros(B, self.num_classes).to(torch.bool)
        eps = -1e4
        for batch_idx in range(B):

            num_pair = num_pairs[batch_idx]

            if num_pair == 0:
                continue

            pair_logit = pair_logits[batch_idx]     # max_num_pairs, 3
            caidx = caidxs[batch_idx]               # max_num_pairs, 3

            flat_logit = pair_logit.view(NUM_TRs*MAX_NUM_PAIRS)[:num_pair*3]
            flat_caidx = caidx.view(NUM_TRs*MAX_NUM_PAIRS)[:num_pair*3]

            N, = flat_logit.shape
            diag_logit = torch.zeros(N, N).to(device)
            diag_logit[torch.arange(N), torch.arange(N)] = flat_logit       # N, N

            one_hot_ca = F.one_hot(flat_caidx.long(), self.num_classes)     # N, 301

            mapped_logit = torch.matmul(diag_logit, one_hot_ca.float())
            mapped_logit[mapped_logit == 0.0] = eps                         # temp
            mapped_logit = mapped_logit.max(dim=0)[0]               
            mapped_logit[mapped_logit == eps] = 0.0

            tr_outputs[batch_idx] = mapped_logit

            exist_caidxs = one_hot_ca.to(torch.bool).any(dim=0)
            exist_bools[batch_idx] = exist_caidxs

        # combine tr outputs(exist) + non exist outputs
        final_outputs = torch.zeros(B, self.num_classes).to(device)
        final_outputs[exist_bools] = tr_outputs[exist_bools]
        final_outputs[~exist_bools] = non_exist_out[~exist_bools]
                
        logits.update({
            'final_outputs' : final_outputs,
            'TR_logit' : pair_logits,    # B, num_pairs, 3
            'TR_target' : TR_targets,   # B, num_pairs, 3
            'perobj_action' : act_out,
            'perobj_object' : obj_out,
        })

        return logits

