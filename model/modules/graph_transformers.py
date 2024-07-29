import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from model.modules.classifiers import MLP
from model.modules.misc import TimeEncode

class SpatioTemporalGraphTransformer_W_OOVD(nn.Module):
    """
        NOTE
    """
    def __init__(
            self,
            device,
            input_dim,
            hidden_dim,

            encoder_layers,
            encoder_heads,
            encoder_use_cache,
            encoder_max_position_embeddings,

            decoder_layers,
            decoder_heads,
            decoder_use_cache,

            **kwargs,
        ):
        super().__init__()
        
        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # cater dataset has a fixed 300 frames, extendable if needed
        self.num_frame = 300

        # dimension convert mlp
        self.input_to_hidden_dim = MLP(
            in_features=input_dim,
            out_features=hidden_dim,
            nlayers=2
        ) if input_dim != hidden_dim else nn.Identity()
        
        # stgt
        self.encoder = transformers.BertModel(
                                        transformers.BertConfig(
                                            hidden_size=hidden_dim,
                                            num_hidden_layers=encoder_layers,
                                            num_attention_heads=encoder_heads,
                                            max_position_embeddings=encoder_max_position_embeddings,
                                            use_cache=encoder_use_cache
                                        )
                                    )
        
        # oovd
        self.object_oriented_video_encoder = transformers.GPT2Model(
                                        transformers.GPT2Config(
                                            n_embd=hidden_dim,
                                            n_layer=decoder_layers,
                                            n_head=decoder_heads,
                                            use_cache=decoder_use_cache
                                        )
                                    )
        
    def forward(self, inp, data, outputs):

        # retreive requirements
        B, NC, MAX_TOK_LEN, _ = inp.shape
        MAX_OBJS_IN_BATCH = max(data['input']['num_objs'])
        pad_mask = data['input']['pad_mask'].to(self.device)            # B, NC, MAX_TOK_LEN
        each_obj_bool = data['input']['each_obj_bool'].to(self.device)  # B, MAX_OBJS_IN_BATCH, NC*MAX_TOK_LEN
        obj_mask = data['input']['obj_mask'].to(self.device)            # B, MAX_OBJS_IN_BATCH

        # stgt's input
        inp = self.input_to_hidden_dim(inp)                             # B, NC, MAX_TOK_LEN, dim
        inp = inp.view(B*NC, MAX_TOK_LEN, self.hidden_dim)              # B*NC, MAX_TOK_LEN, dim
        pad_mask = pad_mask.view(B*NC, MAX_TOK_LEN)                     # B*NC, MAX_TOK_LEN

        # spatiotemporal graph transformer
        clip_repr = self.encoder(
            inputs_embeds=inp,
            attention_mask=pad_mask
        ).last_hidden_state                                             # B*NC, MAX_TOK_LEN, dim

        clip_repr = clip_repr.view(B, 1, NC*MAX_TOK_LEN, self.hidden_dim)
        clip_repr = clip_repr.expand(B, MAX_OBJS_IN_BATCH, NC*MAX_TOK_LEN, self.hidden_dim)
            
        obj_repr = torch.zeros((B, MAX_OBJS_IN_BATCH, self.num_frame, self.hidden_dim)).to(self.device)

        # object-wise rearrangement
        for batch_idx in range(B):
            for obj_idx in range(MAX_OBJS_IN_BATCH):
                # object selection mask
                cur_sel_bool = each_obj_bool[batch_idx, obj_idx]        # NC*MAX_TOK_LEN
                # encoder output 
                cur_reprs = clip_repr[batch_idx, obj_idx]               # NC*MAX_TOK_LEN, dim
                cur_obj_repr = cur_reprs[cur_sel_bool == True]
                obj_repr[batch_idx, obj_idx, :len(cur_obj_repr), :] = cur_obj_repr

        # object-oriented video encoder
        final_output = self.object_oriented_video_encoder(
            inputs_embeds=obj_repr.view(B*MAX_OBJS_IN_BATCH, self.num_frame, self.hidden_dim),
            attention_mask=obj_mask.view(B*MAX_OBJS_IN_BATCH, 1).expand(B*MAX_OBJS_IN_BATCH, self.num_frame)
        ).last_hidden_state

        final_output = final_output[:, -1, :]
        final_output = final_output.view(B, MAX_OBJS_IN_BATCH, self.hidden_dim)
        
        # output
        outputs.update({
            'obj_mask' : obj_mask,
            'transformer_output' : final_output,                                     # B, MAX_OBJS_IN_BATCH, D
        })
        return outputs

class SpatioTemporalGraphTransformer_WO_OOVD(nn.Module):
    """
        NOTE
    """
    def __init__(
            self,
            device,

            input_dim,
            hidden_dim,

            encoder_layers,
            encoder_heads,
            encoder_use_cache,
            encoder_max_position_embeddings,

            **kwargs,
        ):
        super().__init__()
        
        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # dimension convert mlp
        self.input_to_hidden_dim = MLP(
            in_features=input_dim,
            out_features=hidden_dim,
            nlayers=2
        ) if input_dim != hidden_dim else nn.Identity()
        
        # stgt
        self.encoder = transformers.BertModel(
                                        transformers.BertConfig(
                                            hidden_size=hidden_dim,
                                            num_hidden_layers=encoder_layers,
                                            num_attention_heads=encoder_heads,
                                            max_position_embeddings=encoder_max_position_embeddings,
                                            use_cache=encoder_use_cache
                                        )
                                    )
        
        # classification token
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))
        
    def forward(self, inp, data, outputs):

        # retreive requirements
        B, NC, MAX_TOK_LEN, _ = inp.shape
        pad_mask = data['input']['pad_mask'].to(self.device)            # B, NC, MAX_TOK_LEN
        nc_mask = data['input']['nc_mask'].to(self.device)

        # stgt's input
        inp = self.input_to_hidden_dim(inp)                             # B, NC, MAX_TOK_LEN, dim
        inp = inp.view(B*NC, MAX_TOK_LEN, self.hidden_dim)              # B*NC, MAX_TOK_LEN, dim
        inp = torch.cat([self.cls_token.expand(B*NC, 1, self.hidden_dim), inp], dim=1)
        pad_mask = pad_mask.view(B*NC, MAX_TOK_LEN)                     # B*NC, MAX_TOK_LEN
        pad_mask = torch.cat([torch.ones(B*NC, 1).to(pad_mask.device), pad_mask], dim=1)

        # spatiotemporal graph transformer
        clip_repr = self.encoder(
            inputs_embeds=inp,
            attention_mask=pad_mask,
        ).last_hidden_state                                             # B*NC, MAX_TOK_LEN, dim

        output = clip_repr[:, 0, :].view(B, NC, self.hidden_dim)        # clsf token selection
        final_output = output * nc_mask.view(B, NC, 1)
        
        # output
        outputs.update({
            'transformer_output' : final_output
        })
        return outputs
