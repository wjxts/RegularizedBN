# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

#from fairseq.modules import CWN


class RelativeEmbedding(nn.Module):
    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen].
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:   #将embedding个数设多一点，不要出现这种情况
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self._float_tensor)
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(input.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeSinusoidalPositionalEmbedding(RelativeEmbedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1536):
        """

        :param embedding_dim: 每个位置的dimension
        :param padding_idx:
        :param init_size:
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        weights = self.get_embedding(
            init_size+1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb

class MultiheadAttentionSimple(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    __count = 0
    def update_cnt(self):
        MultiheadAttentionSimple.__count += 1
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        penalty=0,
        normalize_q=False,
        normalize_k=False,
        normalize_v=False,
        g0=10,
        fix_g0=False,
    ):
        super().__init__()
        self.id = self.__count  #对每个attention设定唯一标识
        self.update_cnt()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = True
        self.penalty = penalty
        self.normalize_q = normalize_q
        self.normalize_k = normalize_k
        self.normalize_v = normalize_v
        self.fix_g0 = fix_g0
        self.eps = 1e-5
        if self.normalize_q or self.normalize_k:
            self.g0 = Parameter(torch.Tensor(1))
            nn.init.constant_(self.g0, g0)
            if self.fix_g0:
                self.g0 = self.g0.cuda().detach()

        assert self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        #self.k_proj = CWN(self.kdim, embed_dim, NScale = -1, adjustScale=1)
        #self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        #self.q_proj = CWN(embed_dim, embed_dim, NScale = -1, adjustScale=1)
        self.out_proj =nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()  #注意在使用CWN时要去掉！！！

        self.pos_embed = RelativeSinusoidalPositionalEmbedding(self.head_dim, 0, 1200)

        self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(num_heads, self.head_dim)))
        self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(num_heads, self.head_dim)))


    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2)) #weight与weight_v一样
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def orth_loss(self, attention_score):
        #感觉计算量太大了
        #需要额外的mask,mask住的位置可能为-inf,不能直接乘
        #代码不完善
        B, T, _ = attention_score.shape
        cov = torch.bmm(attention_score.transpose(1,2),attention_score)
        cur_loss = 0
        #print("orth loss:",cur_loss)
        return cur_loss

    def norm_irrelevant_attention_weight(self,q,k):
        #q: B, tgt_len, head_dim
        #k: B, src_len, head_dim
        q = q/(q.norm(dim=-1,keepdim=True)+self.eps)
        k = k/(k.norm(dim=-1,keepdim=True)+self.eps)
        attn_weights = torch.bmm(q,k.transpose(1,2))
        return attn_weights

    def loss(self):
        loss = 0
        assert self.training==True, "wrongly adding attention loss!"
        if self.self_attention and self.id<6 and self.penalty>0:
            loss += self.penalty*self.orth_loss(self.attention_score)
        return loss

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        
        assert key is not None and value is not None
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        if not self.normalize_q and not self.normalize_k:
            q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1) #b*n, l,d
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) #b*n, l,d
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) #b*n, l,d
        src_len = k.size(1)
        if self.id<6:
            if self.normalize_q:
                q = q/(q.norm(dim=-1,keepdim=True)+self.eps)
                q *= self.g0
            if self.normalize_k:
                k = k/(k.norm(dim=-1,keepdim=True)+self.eps)
            if self.normalize_v:
                v = v/(v.norm(dim=-1,keepdim=True)+self.eps)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        pos_embed = self.pos_embed(key_padding_mask)  # l x head_dim

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q.view(bsz, self.num_heads, tgt_len, self.head_dim), pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        E_ = torch.einsum('bnqd,ld->bnql', k.view(bsz, self.num_heads, tgt_len, self.head_dim), pos_embed)  # bsz x head x max_len x 2max_len, key对relative的bias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        BDE = BDE.contiguous().view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_weights += BDE*self.scaling

        if attn_mask is not None:
            #print(attn_mask);print(attn_mask.shape);print(attn_weights.shape);exit()
            #attn_mask = attn_mask.repeat(attn_weights.size(0),1,1)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)

        self.attention_score = attn_weights #for addtional loss


        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2

        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD
    
    def _transpose_shift(self, E):
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

        转换为
          0  -10   -200
          1   00   -100
          2   10    000


        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len)*2+1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1,-2)  # bsz x n_head x max_len x max_len

        return E



