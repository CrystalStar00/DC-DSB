
# from itertools import repeat
import itertools
from typing import Union, Optional,List,Any,Literal, Dict,Type
from numpy import ndarray as Array
from torch import Tensor, nn
from inspect import getfullargspec,isfunction
from einops import repeat,rearrange
from pprint import pformat
from torch import einsum
from torch_geometric.nn import SGConv
from torch_geometric.utils import scatter
import warnings
from contextlib import contextmanager
from functools import partial

import anndata as ad
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


from torch import Tensor
from omegaconf import OmegaConf


def instantiate_module(
        cls: Type,
        config: Dict[str, Any],
        extra_kwargs: Dict[str, Any] = None,
        _catch_conflict: bool = True
) -> Any:

    extra_kwargs = extra_kwargs or {}

    if common_keys := sorted(set(config) & set(extra_kwargs)):
        diff_keys = [key for key in common_keys if config[key] != extra_kwargs[key]]

        if diff_keys and _catch_conflict:
            conflicting_config_kwargs = {k: config[k] for k in diff_keys}
            conflicting_extra_kwargs = {k: extra_kwargs[k] for k in diff_keys}
            raise ValueError(
                "There is a conflict between the configuration and the additional parameters. Please resolve the conflict or set _catch_conflict to False.\n"
                f"{conflicting_config_kwargs=}\n"
                f"{conflicting_extra_kwargs=}\n"
            )

    kwargs = {**config, **extra_kwargs}

    try:
        return cls(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Cannot instantiate {cls!r}，The parameters used are as follows:\n{pformat(kwargs)}") from e

def create_activation(name):
    if name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name, n, h=16):
    if name is None:
        return nn.Identity()
    elif name == "layernorm":
        return nn.LayerNorm(n)
    elif name == "batchnorm":
        return nn.BatchNorm1d(n)
    elif name == "groupnorm":
        return nn.GroupNorm(h, n)
    elif name.startswith("groupnorm"):
        inferred_num_groups = int(name.repalce("groupnorm", ""))
        return nn.GroupNorm(inferred_num_groups, n)
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def sinusoidal_embedding(pos: torch.Tensor, dim: int, max_period: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=pos.device)
    args = pos[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        embedding = sinusoidal_embedding(timesteps, dim, max_period)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class Embedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hidden, norm, activation='gelu', dropout=0.,
                 gene_emb=None, fix_embedding=False):
        super().__init__()

        num_genes = len(pretrained_gene_list)
        self.emb = nn.Parameter(torch.randn([num_genes, num_hidden], dtype=torch.float32) * 0.005)
        if fix_embedding:
            self.emb.requires_grad = False
        self.post_layer = nn.Sequential(
            create_activation(activation),
            create_norm(norm, num_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x, pe_input=None, input_gene_list=None, input_gene_idx=None):
        assert pe_input is None  # FIX: deprecate pe_input

        gene_idx = torch.arange(x.shape[1]).long()  # 否则，使用顺序索引
        gene_idx = gene_idx.to(x.device)
        feat = F.embedding(gene_idx, self.emb)
        out = torch.sparse.mm(x, feat)
        out = self.post_layer(out)

        return out, gene_idx

class Encoder(nn.Module):

    def __init__(self,depth,dim,num_heads,dim_head,*,dropout=0.,cond_type='crossattn',cond_cat_input=False):
        super().__init__()

        self.cond_cat_input = cond_cat_input

        if cond_type == 'crossattn':
            self.blocks = nn.ModuleList([
                BasicTransformerBlock(dim, num_heads, dim_head, self_attn=False, cross_attn=True, context_dim=dim,
                                      qkv_bias=True, dropout=dropout, final_act=None)
                for _ in range(depth)])
        elif cond_type == 'mlp':
            self.blocks = nn.ModuleList([
                ConditionEncoderWrapper(nn.Sequential(
                    nn.Linear(dim, dim),
                    "gelu",
                    create_norm("layernorm", dim),
                    nn.Dropout(dropout),
                )) for _ in range(depth)])
        elif cond_type == 'stackffn':
            self.blocks = nn.ModuleList([
                ConditionEncoderWrapper(
                    FeedForward(dim, mult=4, glu=False, dropout=dropout)
                ) for _ in range(depth)])
        else:
            raise ValueError(f'Unknown conditioning type {cond_type!r}')

    def forward(self, x, context_list, cond_emb_list):
        x = x.unsqueeze(1)
        stack = zip(self.blocks, reversed(context_list), reversed(cond_emb_list))
        for i, (blk, ctxt, cond_emb) in enumerate(stack):
            full_cond_emb_list = list(filter(lambda x: x is not None, (ctxt, cond_emb)))
            if self.cond_cat_input:
                full_cond_emb_list.append(x)
            full_cond_emb = torch.cat(full_cond_emb_list, dim=1) if full_cond_emb_list else None
            x = blk(x, context=full_cond_emb)

        return x.squeeze(1)

 # FIX: EmbeddingDict will be refactored to CombinedConditioner


class Decoder(nn.Module):
    def __init__(self, dim, out_dim, dropout=0., norm_type="layernorm", num_layers=1, cond_num_dict=None,
                 cond_emb_dim=None, act="gelu", out_act=None):
        super().__init__()
        if isinstance(act, str) or act is None:
            act = create_activation(act)
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)

        self.cond_num_dict = cond_num_dict
        if self.cond_num_dict is not None:
            cond_emb_dim = cond_emb_dim if cond_emb_dim is not None else dim
            self.cond_embed = EmbeddingDict(cond_num_dict, cond_emb_dim, 1, 1, None)
        else:
            self.cond_embed = None

        self.layers = nn.ModuleList()  # FIX: use MLP layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim),
                act,
                create_norm(norm_type, dim),
                nn.Dropout(dropout),
            ))
        self.layers.append(nn.Sequential(nn.Linear(dim, out_dim), out_act))

    def forward(self, x, conditions=None):
        if conditions is not None:
            cond_emb = self.cond_embed(conditions)[0]
            x = x + cond_emb.squeeze(1)

        for layer in self.layers:
            x = layer(x)

        return x

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)

class ConditionEncoderWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        x = torch.cat((x, context), dim=1).sum(1) if context is not None else x.squeeze(1)
        return self.module(x).unsqueeze(1)

class BatchedOperation:

    def __init__(
        self,
        batch_dim: int = 0,
        plain_num_dim: int = 2,
        ignored_args: Optional[List[str]] = None,
        squeeze_output_batch: bool = True,
    ):
        self.batch_dim = batch_dim
        self.plain_num_dim = plain_num_dim
        self.ignored_args = set(ignored_args or [])
        self.squeeze_output_batch = squeeze_output_batch
        self._is_batched = None

    def __call__(self, func):
        arg_names = getfullargspec(func).args

        def bounded_func(*args, **kwargs):
            new_args = []
            for arg_name, arg in zip(arg_names, args):
                if self.unsqueeze_batch_dim(arg_name, arg):
                    arg = arg.unsqueeze(self.batch_dim)
                new_args.append(arg)

            for arg_name, arg in kwargs.items():
                if self.unsqueeze_batch_dim(arg_name, arg):
                    kwargs[arg_name] = arg.unsqueeze(self.batch_dim)

            out = func(*new_args, **kwargs)

            if self.squeeze_output_batch:
                out = out.squeeze(self.batch_dim)

            return out

        return bounded_func

    def unsqueeze_batch_dim(self, arg_name: str, arg_val: Any) -> bool:
        return (
            isinstance(arg_val, torch.Tensor)
            and (arg_name not in self.ignored_args)
            and (not self.is_batched(arg_val))
        )

    def is_batched(self, val: torch.Tensor) -> bool:
        num_dim = len(val.shape)
        if num_dim == self.plain_num_dim:
            return False
        elif num_dim == self.plain_num_dim + 1:
            return True
        else:
            raise ValueError(
                f"Tensor should have either {self.plain_num_dim} or "
                f"{self.plain_num_dim + 1} number of dimension, got {num_dim}",
            )

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, *, context=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int = 64,
        self_attn: bool = True,
        cross_attn: bool = False,
        ts_cross_attn: bool = False,
        final_act: Optional[nn.Module] = None,
        dropout: float = 0.,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = False,
        qkv_bias: bool = False,
        linear_attn: bool = False,
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
            # assert not (self_attn or linear_attn)
            # attn_cls = TokenSpecificCrossAttention
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention
        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )  # is self-attn if context is none
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )  # is a self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act
        self.checkpoint = checkpoint
        assert not self.checkpoint, 'Checkpointing not available yet'  # FIX: remove checkpiont option

    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:
            x = self.attn1(self.norm1(x), context=context, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), **kwargs) + x
        x = self.ff(self.norm3(x)) + x
        if self.act is not None:
            x = self.act(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out,dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class GEARS_Conditioner(torch.nn.Module):
    def __init__(self, num_perts, out_dim, hidden_size=64, num_go_gnn_layers=1,
                 mlp_layers=2, enable_inference_cache=True, mode="single"):
        super().__init__()

        assert mlp_layers >= 1, f"GEARS MLP layers must be greater than 1, got {mlp_layers}"
        assert mode in ("single", "parallel", "sequential", "mlpparallel", "mlpsequential"), f"Unknown mode {mode!r}"
        assert mode != "mlpsequential" or hidden_size == out_dim, "mlpsequential requires equal hidden and out dim"
        self.mode = mode

        self.num_perts = num_perts
        self.num_layers = num_go_gnn_layers

        # NOTE: we use the first (index 0) embedding as the control embedding
        self.pert_emb = nn.Embedding(num_perts + 1, hidden_size, max_norm=True)
        self.pert_fuse = nn.ModuleList([
            MLP([*([hidden_size] * mlp_layers), out_dim], last_layer_act='ReLU')
            for _ in range(1 if mode == "single" else self.num_layers)])

        self.sim_layers = nn.ModuleList([SGConv(hidden_size, hidden_size, 1)
                                         for _ in range(self.num_layers)])

        self.enable_inference_cache = enable_inference_cache
        self.clear_emb_cache()

    @property
    def use_cache(self) -> bool:
        return self.enable_inference_cache and not self.training

    @property
    def cached_emb(self):
        return self._cached_emb

    def clear_emb_cache(self):
        self._cached_emb = None

    def get_pert_global_emb(self, aug_graph):
        # augment global perturbation embedding with GNN
        G_sim = aug_graph["G_go"]
        G_sim_weight = aug_graph["G_go_weight"]

        pert_global_emb = self.pert_emb.weight
        ctrl_emb = self.pert_emb.weight[0:1]

        pert_global_emb_list = [pert_global_emb]
        for idx, layer in enumerate(self.sim_layers):
            pert_emb = layer(pert_global_emb_list[0 if self.mode == "parallel" else -1][1:], G_sim, G_sim_weight)
            pert_emb = pert_emb if idx == self.num_layers - 1 else pert_emb.relu()
            pert_global_emb_list.append(torch.cat([ctrl_emb, pert_emb], dim=0))

        return pert_global_emb_list[1:]  # skip base embedings

    def forward(self, pert_idx, aug_graph):
        """
        Forward pass of the model
        """
        # NOTE: We use the first embedding as the control embedding and shift
        # everything else by an index of one. We only assign control embedding
        # when all perturbations of the current samlpe are controls.
        pert_index = []
        for idx, i in enumerate(pert_idx.tolist()):
            if all(map(lambda x: x == -1, i)):  # all control -> control
                pert_index.append([idx, 0])
            else:
                pert_index.extend([[idx, j + 1] for j in i if j != -1])
        pert_index = torch.tensor(pert_index, device=pert_idx.device).T

        if self.use_cache:
            # At inference (sampling) time, the global perturbation condition
            # embeddings do not change, so we dont need to recalculate
            if self.cached_emb is None:
                self._cached_emb = [i.detach() for i in self.get_pert_global_emb(aug_graph)]
            pert_global_emb = self.cached_emb

        else:
            self.clear_emb_cache()
            pert_global_emb = self.get_pert_global_emb(aug_graph)

        if self.mode == "single":
            emb = scatter(pert_global_emb[-1][pert_index[1]], pert_index[0], dim=0)
            out = self.pert_fuse[0](emb)
        elif self.mode in ("parallel", "sequential"):
            out = []
            for pert_emb, pert_fuse in zip(pert_global_emb, self.pert_fuse):
                emb = scatter(pert_emb[pert_index[1]], pert_index[0], dim=0)
                out.append(pert_fuse(emb))
        elif self.mode in ("mlpparallel", "mlpsequential"):
            out = [scatter(pert_global_emb[-1][pert_index[1]], pert_index[0], dim=0)]
            for pert_fuse in self.pert_fuse:
                out.append(pert_fuse(out[0 if self.mode == "mlpparallel" else -1]))
            out = out[:-1]
        else:
            raise ValueError(f"Unknown mode {self.mode!r}, should have been caught earlier.")

        return out

class EmbeddingDict(nn.Module):
    TEXT_EMB_DIR = './dataset/ontology_resources'

    def __init__(self, num_embed_dict, embedding_dim, depth, embedding_tokens=1,
                 norm_layer=None, freeze=False, mask_ratio=0.0, text_emb=None,
                 text_emb_file=None, freeze_text_emb=True, text_proj_type='linear',
                 stackfnn_glu_flag=False, text_proj_hidden_dim=512, text_proj_act=None,
                 text_proj_num_layers=2, text_proj_norm=None, text_proj_dropout=0.,
                 gears_flag=False, gears_mode="single", num_perts=None, gears_hidden_size=64,
                 gears_mlp_layers=2, gears_norm=None, num_go_gnn_layers=1):
        super().__init__()
        size = embedding_dim * embedding_tokens
        n = embedding_tokens
        d = embedding_dim

        self.keys = sorted(num_embed_dict)  # ensure consistent ordering
        self.mask_ratio = mask_ratio

        self.emb_dict = nn.ModuleDict()

        for key in self.keys:
            self.emb_dict[key] = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding(
                        num_embed_dict[key],
                        size,
                        _freeze=freeze,
                    ),
                    create_norm(norm_layer, size),
                    Rearrange('b (n d) -> b n d', n=n, d=d),
                )
                for _ in range(depth)
            ])

        # if text_emb is not None or text_emb_file is not None:
        #     if text_emb is None:
        #         text_emb = torch.load(f'{self.TEXT_EMB_DIR}/{text_emb_file}')
        #     if text_proj_type == 'linear':
        #         text_proj = nn.Linear(text_emb.shape[1], size)
        #     elif text_proj_type == 'stackffn':
        #         text_proj = FeedForward(text_emb.shape[1], dim_out=size, mult=4, glu=stackfnn_glu_flag)
        #     elif text_proj_type == 'mlp':
        #         text_proj = MLPLayers(text_emb.shape[1], size, text_proj_hidden_dim, text_proj_num_layers,
        #                               text_proj_dropout, text_proj_norm, text_proj_act)
        #     else:
        #         raise NotImplementedError(f"Unsupported text_proj_type {text_proj_type}")
        #
        #     text_act = create_activation(text_proj_act)
        #     if text_proj_norm is None and norm_layer is not None:
        #         text_norm = create_norm(norm_layer, size)
        #     else:
        #         text_norm = create_norm(text_proj_norm, size)
        #     self.keys.append("text")
        #     self.emb_dict['text'] = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Embedding.from_pretrained(text_emb, freeze=freeze_text_emb),
        #             text_proj,
        #             text_norm,
        #             text_act,
        #             Rearrange('b (n d) -> b n d', n=n, d=d),
        #         )
        #         for _ in range(depth)
        #     ])

        if num_perts is not None and gears_flag:
            self.keys.append('pert')
            self.gears_mode = gears_mode
            gears_kwargs = dict(num_perts=num_perts, out_dim=size, mode=gears_mode,
                                hidden_size=gears_hidden_size, mlp_layers=gears_mlp_layers)
            if gears_mode == "single":
                self.emb_dict['pert'] = nn.ModuleList([
                    nn.Sequential(
                        GEARS_Conditioner(num_go_gnn_layers=num_go_gnn_layers, **gears_kwargs),
                        create_norm(gears_norm, size),
                        Rearrange('b (n d) -> b n d', n=n, d=d),
                    )
                    for _ in range(depth)
                ])
            else:
                self.emb_dict['pert'] = nn.ModuleList([
                    GEARS_Conditioner(num_go_gnn_layers=depth, **gears_kwargs),
                    nn.ModuleList([create_norm(gears_norm, size) for _ in range(depth)]),
                    Rearrange('b (n d) -> b n d', n=n, d=d),
                ])

    def __iter__(self):
        yield from self.keys

    def __getitem__(self, key):
        return self.emb_dict[key]

    def forward(self, input: Dict[str, torch.Tensor], aug_graph=None) -> List[torch.Tensor]:
        # Outer list: condition types; inner list: layer depth 初始化输出列表，用于存储每种特征类型的嵌入结果
        out = []
        for key in self.keys:
            masked_input = input[key].long()
            if (
                isinstance(self[key][0], GEARS_Conditioner)  # single
                or isinstance(self[key][0][0], GEARS_Conditioner)  # parallel | sequential
            ):
                emb_list = []
                if self.gears_mode == "single":
                    for emb in self[key]:
                        gears_out = emb[0](masked_input, aug_graph)
                        emb_list.append(emb[1:](gears_out))
                else:
                    gears_out = self[key][0](masked_input, aug_graph)
                    stack = zip(gears_out, self[key][1], itertools.repeat(self[key][2]))
                    for emb, norm, rearrange in stack:
                        emb_list.append(rearrange(norm(emb)))
            else:
                emb_list = [emb(masked_input) for emb in self[key]]

            out.append(emb_list)

        out = [torch.cat(embs, dim=1) for embs in zip(*out)]

        return out

class TransformerNetwork(nn.Module):
    def __init__(self,pretrained_gene_list,input_gene_list=None, dropout=0.0,encoder_type='stackffn', embed_dim=1024, depth=4, dim_head=64, num_heads=4,
                 decoder_embed_dim=512, decoder_embed_type='linear', decoder_num_heads=4,
                 decoder_dim_head=64, cond_dim=None, cond_tokens=1, cond_type='crossattn', cond_strategy='full_mix',
                 cond_emb_type='linear', cond_num_dict=None, cond_cat_input=False,
                 post_cond_num_dict=None, post_cond_layers=2, post_cond_norm='layernorm',
                 norm_layer='layernorm', mlp_time_embed=False, no_time_embed=False,
                 activation='gelu', stackfnn_glu_flag=False,
                 cond_emb_norm=None, num_perts=None, gears_flag=False, gears_hidden_size=64,
                 gears_mode="single", gears_mlp_layers=2, gears_norm=None, num_go_gnn_layers=1,
                 is_diffusing=False):
        super().__init__()

        activation = create_activation(activation)
        self.in_dim = len(pretrained_gene_list)
        self.pretrained_gene_list = pretrained_gene_list
        self.input_gene_list = input_gene_list
        pretrained_gene_index = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))
        self.input_gene_idx = torch.tensor([
            pretrained_gene_index[o] for o in self.input_gene_list
            if o in pretrained_gene_index
        ]).long() if self.input_gene_list is not None else None
        self.depth = depth
        self.is_diffusing=is_diffusing
        # self.in_dim=in_dim
        # self.out_dim=out_dim

        #embedding encoder
        assert embed_dim == decoder_embed_dim
        full_embed_dim = embed_dim * cond_tokens
        self.post_encoder_layer = Rearrange('b (n d) -> b n d', n=cond_tokens, d=embed_dim)

        self.embedder = Embedder(pretrained_gene_list,full_embed_dim, 'layernorm', dropout=dropout)

        self.encoder_type = encoder_type
        if encoder_type == 'attn':
            self.blocks = nn.ModuleList([
                BasicTransformerBlock(full_embed_dim, num_heads, dim_head, self_attn=True, cross_attn=False,
                                      dropout=dropout, qkv_bias=True, final_act=activation)
                for _ in range(depth)])
        elif encoder_type in ('mlp', 'mlpparallel'):
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(full_embed_dim, full_embed_dim),
                    activation,
                    create_norm(norm_layer, full_embed_dim),
                ) for _ in range(depth)])
        elif encoder_type in ('stackffn', 'ffnparallel'):
            self.blocks = nn.ModuleList([
                # FeedForward(full_embed_dim, mult=4, glu=False, dropout=dropout)
                nn.Sequential(
                    FeedForward(full_embed_dim, mult=4, glu=False, dropout=dropout),
                    create_norm(norm_layer, full_embed_dim),
                ) for _ in range(depth)])
        elif encoder_type == 'none':
            self.blocks = None
        else:
            raise ValueError(f'Unknown encoder type {encoder_type}')


        #MAE decoder
        self.decoder_embed_dim = decoder_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * decoder_embed_dim, decoder_embed_dim),
        ) if mlp_time_embed else nn.Identity()
        self.no_time_embed = no_time_embed

        self.cond_type = cond_type
        assert cond_strategy in ("full_mix", "pre_mix")
        self.cond_strategy = cond_strategy
        self.cond_emb_type = cond_emb_type
        self.cond_tokens = cond_tokens
        self.cond_cat_input = cond_cat_input

        if cond_dim is not None or cond_num_dict is not None:
            if cond_emb_type == 'linear':
                assert cond_dim is not None
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, decoder_embed_dim * cond_tokens),
                    Rearrange('b (n d) -> b n d', n=cond_tokens, d=decoder_embed_dim),
                )
            elif cond_emb_type == 'embedding':
                assert cond_num_dict is not None
                self.cond_embed = EmbeddingDict(cond_num_dict, decoder_embed_dim, depth,cond_tokens,
                                                # text_proj_dropout=dropout, G_go=G_go,
                                                # G_go_weight=G_go_weight, num_perts=num_perts,
                                                text_proj_dropout=dropout, gears_flag=gears_flag, num_perts=num_perts,
                                                gears_hidden_size=gears_hidden_size, gears_mode=gears_mode,
                                                gears_mlp_layers=gears_mlp_layers, gears_norm=gears_norm,
                                                num_go_gnn_layers=num_go_gnn_layers)
            elif cond_emb_type == 'none':
                self.cond_embed = None
            else:
                raise ValueError(f"Unknwon condition embedder type {cond_emb_type}")
        else:
            self.cond_embed = None
        #encoder
        self.encoder = Encoder(depth, decoder_embed_dim, decoder_num_heads, decoder_dim_head,
                               dropout=dropout, cond_type=cond_type, cond_cat_input=cond_cat_input)
        self.decoder_embed_type = decoder_embed_type
        assert decoder_embed_type in ['linear', 'embedder', 'encoder']
        if decoder_embed_type == 'linear':
            self.decoder_embed = nn.Linear(self.in_dim, decoder_embed_dim)
        elif decoder_embed_type == 'embedder':
            self.decoder_embed = Embedder(pretrained_gene_list, decoder_embed_dim, 'layernorm', dropout=dropout)
        elif decoder_embed_type == 'encoder':
            self.decoder_embed = self.embedder

        self.decoder_norm = create_norm(norm_layer, decoder_embed_dim)
        self.decoder = Decoder(decoder_embed_dim, self.in_dim, dropout, post_cond_norm,
        post_cond_layers, post_cond_num_dict, act = activation,cond_emb_dim = decoder_embed_dim)
        # --------------------------------------------------------------------------

        self.initialize_weights()

        def initialize_weights(self):
            # initialize linear and normalization layers
            self.apply(init_weights)

        def forward_encoder(self, x, pe_input=None, input_gene_list=None, input_gene_idx=None):
            x, gene_idx = self.embedder(x)

            if self.encoder_type in ("mlpparallel", "ffnparallel"):
                hist = [self.post_encoder_layer(blk(x)) for blk in self.blocks]
            else:
                hist = []
            for blk in self.blocks:  # apply context encoder blocks
                x = blk(x)
                hist.append(self.post_encoder_layer(x))
            return hist, gene_idx

    def initialize_weights(self):
        # initialize linear and normalization layers
        self.apply(init_weights)

    def forward_encoder(self, x, pe_input=None, input_gene_list=None, input_gene_idx=None):
        x, gene_idx = self.embedder(x)

        if self.encoder_type in ("mlpparallel", "ffnparallel"):
            hist = [self.post_encoder_layer(blk(x)) for blk in self.blocks]
        else:
            hist = []
        for blk in self.blocks:  # apply context encoder blocks
            x = blk(x)
            hist.append(self.post_encoder_layer(x))
        return hist, gene_idx

    def forward_decoder(self, x, context_list, timesteps=None, pe_input=None, conditions=None,
                        input_gene_list=None, input_gene_idx=None, aug_graph=None,
                        return_latent=False):
        # embed tokens
        if self.decoder_embed_type == 'linear':
            x = self.decoder_embed(x)
        else:
            input_gene_list = default(input_gene_list, self.input_gene_list)
            input_gene_idx = default(input_gene_idx, self.input_gene_idx)
            x, _ = self.decoder_embed(x, pe_input, input_gene_list, input_gene_idx)

        # calculate time embedding
        if timesteps is not None and not self.no_time_embed:
            timesteps = timesteps.repeat(x.shape[0]) if len(timesteps) == 1 else timesteps
            time_embed = self.time_embed(timestep_embedding(timesteps, self.decoder_embed_dim))
            x = x + time_embed
            # x = torch.cat([x, time_embed], dim=0)

        # calculate cell condition embedding
        cond_emb_list = None if conditions is None else self.cond_embed(conditions, aug_graph=aug_graph)
        if not isinstance(cond_emb_list, list):
            cond_emb_list = [cond_emb_list] * self.depth
        x = self.encoder(x, context_list, cond_emb_list)
        x = self.decoder_norm(x)
        return x if return_latent else self.decoder(x, conditions)


    def forward(self, x_orig, x, timesteps=None, pe_input=None, conditions=None, input_gene_list=None,
                input_gene_idx=None, target_gene_list=None, aug_graph=None):
        context_list, gene_idx = self.forward_encoder(x_orig, pe_input, input_gene_list, input_gene_idx)
        pred = self.forward_decoder(x, context_list, timesteps, pe_input, conditions, input_gene_list,
                                    input_gene_idx, aug_graph=aug_graph)

        return pred

