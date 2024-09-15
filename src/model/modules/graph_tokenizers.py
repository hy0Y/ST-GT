import torch
import torch.nn as nn

from model.modules.misc import TimeEncode

class CaterGraphTokenizer(nn.Module):
    def __init__(
            self,
            device,

            # each feature's input dim
            inp_attr_dim,
            inp_coor_dim,
            inp_time_dim,
            inp_n_id_dim,
            inp_t_id_dim,

            # each feature's final dim
            out_attr_dim,
            out_coor_dim,
            out_time_dim,
            out_n_id_dim,
            out_t_id_dim,

            # concatenated dim
            out_dim,

            **kwargs,
        ):
        super().__init__()

        self.device = device
        self.out_dim = out_dim

        # input feature's dimension
        self.inp_attr_dim = inp_attr_dim
        self.inp_coor_dim = inp_coor_dim
        self.inp_time_dim = inp_time_dim
        self.inp_n_id_dim = inp_n_id_dim
        self.inp_t_id_dim = inp_t_id_dim

        # output each feature's dimension
        self.out_attr_dim = out_attr_dim
        self.out_coor_dim = out_coor_dim
        self.out_time_dim = out_time_dim
        self.out_n_id_dim = out_n_id_dim
        self.out_t_id_dim = out_t_id_dim

        # feature encoders
        self.attr_enc = nn.Linear(2*inp_attr_dim, out_attr_dim) if 2*inp_attr_dim != out_attr_dim else nn.Identity()
        self.coor_enc = nn.Linear(2*inp_coor_dim, out_coor_dim) if 2*inp_coor_dim != out_coor_dim else nn.Identity()
        self.time_enc = TimeEncode(expand_dim=out_time_dim//2)
        self.n_id_enc = nn.Identity()

        # type identifiers
        self.type_emb = nn.Embedding(
            num_embeddings=3,   # node, spatial edge, temporal edge
            embedding_dim=out_dim
        )

        self.kwargs = kwargs

    def forward(self, data):
                
        # inputdata unpacking
        inp = data['input']
        tok_n_idxs = inp['token_pair_idx'].to(self.device)
        tok_t_idxs = inp['token_pair_time'].to(self.device)
        tok_types = inp['token_types'].to(self.device)


        # shape_idxs = input['shape_idx']
        attr_lookup = inp['attr_feats_lookup'].to(self.device)
        coor_feats = inp['coord_feats'].to(self.device)
        idx_in_lookup = inp['idx_in_lookup'].to(self.device)
        n_id_lookup = inp['n_id_lookup'].to(self.device)

        # required shape
        B, NC, MAX_TOK_LEN, _  = tok_n_idxs.shape # [B, num_clips, token_len, dim]
        _, MAX_OBJS_TOTAL, attr_dim = attr_lookup.shape
        _, _, N_ID_DIM, _ = n_id_lookup.shape

        # attribute
        attr_lookup = attr_lookup.view(B, 1, MAX_OBJS_TOTAL, self.inp_attr_dim).expand(B, NC, MAX_OBJS_TOTAL, self.inp_attr_dim)
        tok_n_idxs = tok_n_idxs.view(B, NC, MAX_TOK_LEN*2, 1).expand(B, NC, MAX_TOK_LEN*2, self.inp_attr_dim)
        attr_feats = torch.gather(attr_lookup, 2, tok_n_idxs.long()).view(B, NC, MAX_TOK_LEN, 2*self.inp_attr_dim)
        
        # coordinate
        coor_feats = coor_feats # B, NC, MAX_TOK, 24

        # time
        tok_t_idxs = tok_t_idxs.view(B*NC, MAX_TOK_LEN*2)
        max_time_in_clip = tok_t_idxs.max(dim=-1)[0].view(B*NC, 1).expand(B*NC, MAX_TOK_LEN*2)
        phase_in_clip = max_time_in_clip - tok_t_idxs   # B*NC, MAX_TOK_LEN*2
        time_feats = self.time_enc(phase_in_clip).view(B, NC, MAX_TOK_LEN, self.out_time_dim)

        # node identifier
        n_id_lookup = n_id_lookup.view(B, NC, N_ID_DIM, N_ID_DIM)
        idx_in_lookup = idx_in_lookup.view(B, NC, MAX_TOK_LEN*2, 1).expand(B, NC, MAX_TOK_LEN*2, N_ID_DIM)
        n_id_feats = torch.gather(n_id_lookup, 2, idx_in_lookup.long()).view(B, NC, MAX_TOK_LEN, 2*N_ID_DIM)

        # type identifier
        t_id_feats = self.type_emb(tok_types.long())

        # encode each feature
        attr_feats = self.attr_enc(attr_feats.type(torch.float32))
        coor_feats = self.coor_enc(coor_feats)
        time_feats = time_feats
        n_id_feats = self.n_id_enc(n_id_feats)

        # concatenate and make token
        tokens = []
        tokens.append(attr_feats)
        tokens.append(coor_feats)
        tokens.append(time_feats)
        tokens.append(n_id_feats)
        input_tokens = torch.cat(tokens, dim=-1)        # B, NC, MAX_TOK_LEN, 1112

        input_tokens = input_tokens + t_id_feats

        return input_tokens


class MomaGraphTokenizer(nn.Module):
    def __init__(
            self,
            device,

            # each feature's input dim
            inp_attr_dim,
            inp_bbox_dim,
            inp_time_dim,
            inp_n_id_dim,
            inp_t_id_dim,

            # each feature's final dim
            out_attr_dim,
            out_bbox_dim,
            out_time_dim,
            out_n_id_dim,
            out_t_id_dim,

            # concatenated dim
            out_dim,

            **kwargs,
        ):
        super().__init__()

        self.device = device
        self.out_dim = out_dim

        # input feature's dimension
        self.inp_attr_dim = inp_attr_dim
        self.inp_bbox_dim = inp_bbox_dim
        self.inp_time_dim = inp_time_dim
        self.inp_n_id_dim = inp_n_id_dim
        self.inp_t_id_dim = inp_t_id_dim

        # output each feature's dimension
        self.out_attr_dim = out_attr_dim
        self.out_bbox_dim = out_bbox_dim
        self.out_time_dim = out_time_dim
        self.out_n_id_dim = out_n_id_dim
        self.out_t_id_dim = out_t_id_dim

        # feature encoders
        self.attr_enc = nn.Linear(inp_attr_dim, out_attr_dim)
        self.bbox_enc = nn.Linear(2*inp_bbox_dim, out_bbox_dim)
        self.time_enc = TimeEncode(expand_dim=out_time_dim//2)
        self.n_id_enc = nn.Linear(2*inp_n_id_dim, out_n_id_dim)

        # type identifiers
        self.type_emb = nn.Embedding(
            num_embeddings=3,       # node, spatial edge, temporal edge
            embedding_dim=out_dim
        )

        self.kwargs = kwargs

    def forward(self, data):
        
        # inputdata unpacking
        inp = data['input']
        num_objs = inp['num_objs']
        tok_n_idxs = inp['token_pair_idx'].to(self.device)
        tok_t_idxs = inp['token_pair_time'].to(self.device)
        tok_types = inp['token_types'].to(self.device)

        tok_e_idxs = inp['token_eidx'].to(self.device)

        nfeats_lup = inp['nfeats_lup'].to(self.device)
        efeats_lup = inp['efeats_lup'].to(self.device)
        bbox_feats = inp['bbox_feats'].to(self.device)

        idx_in_lookup = inp['idx_in_lookup'].to(self.device)
        n_id_lookup = inp['n_id_lookup'].to(self.device)

        # required dim
        B, NC, MAX_TOK_LEN, _  = tok_n_idxs.shape # [batch size, num_clips, token_len, 2]
        NFEAT_DIM,EFEAT_DIM = nfeats_lup.shape[-1], efeats_lup.shape[-1]
        _, _, N_ID_DIM, _ = n_id_lookup.shape

        MAX_OBJS_TOTAL = max(num_objs) + 1

        # n feat
        nfeats_lup = nfeats_lup.view(B, 1, MAX_OBJS_TOTAL, NFEAT_DIM).expand(B, NC, MAX_OBJS_TOTAL, NFEAT_DIM)
        single_tok_n_idxs = tok_n_idxs[:, :, :, :1]
        single_tok_n_idxs = single_tok_n_idxs.view(B, NC, MAX_TOK_LEN, 1).expand(B, NC, MAX_TOK_LEN, NFEAT_DIM)
        nfeats = torch.gather(nfeats_lup, 2, single_tok_n_idxs.long()).view(B, NC, MAX_TOK_LEN, NFEAT_DIM)
        NONEDGE = ((tok_types == 0) | (tok_types == 2)).view(B, NC, MAX_TOK_LEN, 1).expand(B, NC, MAX_TOK_LEN, NFEAT_DIM)   # 2, 12, maxtoklen, 384
        nfeats[NONEDGE == False] = 0     # 2, 12, 297, 384

        _, MAX_EDGES, _ = efeats_lup.shape

        # e feat
        efeats_lup = efeats_lup.view(B, 1, MAX_EDGES, EFEAT_DIM).expand(B, NC, MAX_EDGES, EFEAT_DIM)
        tok_e_idxs = tok_e_idxs.view(B, NC, MAX_TOK_LEN, 1).expand(B, NC, MAX_TOK_LEN, EFEAT_DIM)
        efeats = torch.gather(efeats_lup, 2, tok_e_idxs.long()).view(B, NC, MAX_TOK_LEN, EFEAT_DIM)
        efeats[NONEDGE == True] = 0

        attr_feats = torch.zeros_like(nfeats).to(nfeats.device)
        attr_feats[NONEDGE] = nfeats[NONEDGE]
        attr_feats[~NONEDGE] = efeats[~NONEDGE].to(torch.float64)

        # coordinate
        bbox_feats = bbox_feats # B, NC, MAX_TOK, 24

        # time
        tok_t_idxs = tok_t_idxs.view(B*NC, MAX_TOK_LEN*2)
        max_time_in_clip = tok_t_idxs.max(dim=-1)[0].view(B*NC, 1).expand(B*NC, MAX_TOK_LEN*2)
        phase_in_clip = max_time_in_clip - tok_t_idxs   # B*NC, MAX_TOK_LEN*2
        time_feats = self.time_enc(phase_in_clip).view(B, NC, MAX_TOK_LEN, self.out_time_dim)
        
        # node identifier
        n_id_lookup = n_id_lookup.view(B, NC, N_ID_DIM, N_ID_DIM)
        idx_in_lookup = idx_in_lookup.view(B, NC, MAX_TOK_LEN*2, 1).expand(B, NC, MAX_TOK_LEN*2, N_ID_DIM)
        n_id_feats = torch.gather(n_id_lookup, 2, idx_in_lookup.long()).view(B, NC, MAX_TOK_LEN, 2*N_ID_DIM)
        padded_n_id_feats = torch.zeros(B, NC, MAX_TOK_LEN, self.inp_n_id_dim*2).to(n_id_feats.device)
        padded_n_id_feats[:, :, :, :n_id_feats.shape[-1]] = n_id_feats

        # type identifier
        t_id_feats = self.type_emb(tok_types.long())

        # encode each feature
        attr_feats = self.attr_enc(attr_feats.type(torch.float32))
        bbox_feats = self.bbox_enc(bbox_feats)
        time_feats = time_feats
        n_id_feats = self.n_id_enc(padded_n_id_feats)

        # concatenate and make token
        tokens = []
        tokens.append(attr_feats)
        tokens.append(bbox_feats)
        tokens.append(time_feats)
        tokens.append(n_id_feats)
        input_tokens = torch.cat(tokens, dim=-1)        # B, NC, MAX_TOK_LEN, 1112

        input_tokens = input_tokens + t_id_feats

        return input_tokens