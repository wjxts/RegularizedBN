# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.serialization import save
from fairseq import utils
from fairseq.modules import LayerNorm


from fairseq.modules.norm.mask_identity import MaskIdentityNorm as InFcNorm

from fairseq.modules.norm.mask_identity import MaskIdentityNorm as NoNorm

#from fairseq.modules import CWN
from fairseq.modules import NormSelect
from fairseq.modules import FcSelect
from fairseq.modules import ActivationSelect
from fairseq.modules import MultiheadAttention, MultiheadAttentionSimple
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
import numpy as np
from scipy import io
import time 



from .statistics_utils import insert_probe
from .statistics_utils import record_forward_weight_norm
from .statistics_utils import save_attn_weights
from .statistics_utils import save_residual_proportion

from .statistics_init import init_config

cond_name = ['c_max', 'c_20', 'c_50', 'c_80', 'r_total', 'r',  'intra_average_sim', 'inter_average_sim', 'total_average_sim']

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    __count = 0
    def update_cnt(self):
        TransformerEncoderLayer.__count += 1

    def forward_hook(self,savefile):
        def forward_hook_template(m,i,o):
            input = i[0]
            self.input_norm[savefile].append(torch.norm(input,dim=(0,2)))
        return forward_hook_template

    def __init__(self, args):
        super().__init__()
        self.id = self.__count  #对每个transformer_encoder设定唯一标识
        self.update_cnt()
        
        
        self.orth_penalty = args.orth_penalty
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        #self.self_attn_layer_norm = NormSelect(args.encoder_att_norm,self.embed_dim, args.wseq, prefix = self.prefix)
        
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        #self.dropout_module = FairseqDropout(args.encoder_dropout, module_name=self.__class__.__name__)
        #self.after_norm_dropout_module = nn.Dropout(p=args.encoder_after_norm_dropout) 

        #self.activation_fn = utils.get_activation_fn(
        #    activation=getattr(args, "activation_fn", "relu")
        #)
        self.activation_fn = ActivationSelect(args.encoder_activation)
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        #self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        #self.fc2 = nn.Linear(args.encoder_ffn_embed_dim,self.embed_dim)
        self.fc1 = FcSelect(self.embed_dim, args.encoder_ffn_embed_dim, args.encoder_fc1, self.id)
        
        self.fc2 = FcSelect(args.encoder_ffn_embed_dim, self.embed_dim, args.encoder_fc2, self.id)
        
        init_config(self, args)
        self.self_attn_layer_norm = NormSelect(args.encoder_att_norm, self.embed_dim, self.id, self.prefix)
        
        self.in_ffn_norm = NormSelect(args.encoder_in_ffn_norm, args.encoder_ffn_embed_dim, self.id, self.prefix)
       
        #self.before_relu_nonorm = NormSelect("batch_nonorm", args.encoder_ffn_embed_dim, self.id, self.prefix)
        
        #self.after_relu_nonorm = NormSelect("batch_nonorm", args.encoder_ffn_embed_dim, self.id, self.prefix)
        
        self.final_layer_norm = NormSelect(args.encoder_ffn_norm, self.embed_dim, self.id, self.prefix)

        
        #m.register_parameter('weight_g',nn.parameter.Parameter(torch.ones(3,1)))可更改scale
        if self.id == -1:
            self.fc1.weight_v.register_hook(self.hook_v)
            self.fc1.weight_g.register_hook(self.hook_v)
        
    def orth_loss(self, w):
        d1,d2 = w.shape
        I = torch.eye(min(d1,d2)).cuda()
        if d1>d2:
            cov = torch.matmul(w.transpose(0,1),w)
        else:
            cov = torch.matmul(w,w.transpose(0,1))
        cur_loss = ((cov-2*I)**2).sum()
        #print("orth loss:",cur_loss)
        return cur_loss

    def loss(self):
        assert self.training, "wrongly adding orthorgonal penalty!"
        loss = 0
        if self.orth_penalty>0:
            loss += self.orth_penalty*self.orth_loss(self.fc1.weight)
            loss += self.orth_penalty*self.orth_loss(self.fc2.weight)
        return loss

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        if args.encoder_attention=='origin':
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                normalize_q=args.normalize_q,
                normalize_k=args.normalize_k,
                normalize_v=args.normalize_v,
                g0=args.g0,
                fix_g0=args.fix_g0,
            )
        elif args.encoder_attention=='position':
            return MultiheadAttentionSimple(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                normalize_q=args.normalize_q,
                normalize_k=args.normalize_k,
                normalize_v=args.normalize_v,
                g0=args.g0,
                fix_g0=args.fix_g0,
            )
        else:
            print("error encoder attention!");exit()

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, src_tokens=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        #if(encoder_padding_mask.sum()!=0 and 0): #有时前边几个为True，剩余为False
        #    print("has mask!",self.id)
        #    print(encoder_padding_mask.sum())
        #    print(encoder_padding_mask)
            #exit()
        #if(attn_mask is not None and 0):  #均为None
        #    print("has att mask!",self.id)
        #    print(attn_mask);exit()
        

        if self.training:
            self.step += 1
            self.src_tokens = src_tokens
            self.mask = 1 - encoder_padding_mask.unsqueeze(dim=-1).type_as(x)

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        #record the weight matrix in FFN module
        record_forward_weight_norm(self)

        residual = x
        if self.normalize_before:
            position = 'att_norm_input'
            insert_probe(self, x, position)
           
            x = self.self_attn_layer_norm(x,encoder_padding_mask)

            #x = self.after_norm_dropout_module(x)
            
        #Attention Layer
        position = 'att_input'
        insert_probe(self, x, position)

        x, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        
        save_attn_weights(self, attn_weights)

        position = 'att_output'
        insert_probe(self, x, position)

        x = self.dropout_module(x)  #先子模块dropout，再与residual相加

        
        save_residual_proportion(self, x, residual, module='att')

        x = residual + x
        if not self.normalize_before:
            position = 'att_norm_input'
            insert_probe(self, x, position)
            
            x = self.self_attn_layer_norm(x, encoder_padding_mask)

            #x = self.after_norm_dropout_module(x)
            
        #FFN
        residual = x
        if self.normalize_before:
            position = 'ffn_norm_input'
            insert_probe(self, x, position)
        
            x = self.final_layer_norm(x,encoder_padding_mask)

            #x = self.after_norm_dropout_module(x)


        position = 'ffn_input'
        insert_probe(self, x, position)

        x = self.fc1(x)

        position = 'before_relu'
        insert_probe(self, x, position)

        x = self.in_ffn_norm(x, encoder_padding_mask)

        #x = self.before_relu_nonorm(x, encoder_padding_mask)

        x = self.activation_fn(x)

        #x = self.after_relu_nonorm(x, encoder_padding_mask)

        position = 'after_relu'
        insert_probe(self, x, position)

        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        position = 'ffn_output'
        insert_probe(self, x, position)

        x = self.dropout_module(x)

        save_residual_proportion(self, x, residual, module='ffn')

        x = residual + x
        if not self.normalize_before:
            position = 'ffn_norm_input'
            insert_probe(self, x, position)
            x = self.final_layer_norm(x, encoder_padding_mask)

            #x = self.after_norm_dropout_module(x)
    
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    __count = 0
    def update_cnt(self):
        TransformerDecoderLayer.__count += 1
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.id = self.__count  #对每个transformer_encoder设定唯一标识
        self.update_cnt()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        #self.dropout_module = FairseqDropout(args.decoder_dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        #self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        #self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.self_attn_layer_norm = NormSelect(args.decoder_att_norm ,self.embed_dim)
        #self.self_attn_layer_norm = NoNorm(self.embed_dim)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            #self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            #self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
            #self.encoder_attn_layer_norm = NoNorm(self.embed_dim)
            self.encoder_attn_layer_norm = NormSelect(args.decoder_cross_att_norm ,self.embed_dim)
        self.fc1 = FcSelect(self.embed_dim, args.decoder_ffn_embed_dim, args.decoder_fc1)
        self.in_ffn_norm = NormSelect(args.decoder_in_ffn_norm, args.decoder_ffn_embed_dim)
        self.fc2 = FcSelect(args.decoder_ffn_embed_dim, self.embed_dim, args.decoder_fc2)
        #self.fc1 = self.build_fc1(
        #    self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        #)
        #self.fc2 = self.build_fc2(
        #    args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        #)
        #self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        #self.final_layer_norm = LayerNorm(self.embed_dim)
        #self.final_layer_norm = NoNorm(self.embed_dim)
        self.final_layer_norm = NormSelect(args.decoder_ffn_norm ,self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        #if self.id==0:
        #    print("decoding!")
        if need_head_weights:
            need_attn = True
        residual = x
        if self.normalize_before:
            #print(self_attn_padding_mask)
            #assert self_attn_padding_mask is not None, "wrong attn padding mask!" #为什么训练时会出现none?
            x = self.self_attn_layer_norm(x, self_attn_padding_mask)
        if prev_self_attn_state is not None: #false
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ): #false
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x, self_attn_padding_mask)

        if self.encoder_attn is not None: #true
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x, self_attn_padding_mask)
            if prev_attn_state is not None: #false
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x, self_attn_padding_mask)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x, self_attn_padding_mask)

        x = self.fc1(x)
        x = self.in_ffn_norm(x,self_attn_padding_mask)  #add decoder padding mask?
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x, self_attn_padding_mask)
        if self.onnx_trace and incremental_state is not None: #false
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
