import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from einops import rearrange, repeat
import os
from .dynamic_mlp import *
from .geomf_modules import GSCM, HAFM
from types import SimpleNamespace
import math
import copy
try:
    from mamba_util import PatchMerging,SimplePatchMerging, Stem, SimpleStem, Mlp
except:
    from .mamba_util import PatchMerging, SimplePatchMerging, Stem, SimpleStem, Mlp
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
torch.set_printoptions(threshold=torch.inf)
class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])



class ConvFFN(nn.Module):

    def __init__(self, channels, expansion=2, drop=0.0):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H, W):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=False,
        d_state = 64,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        #convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # modified from RMSNormGated to layer norm
        #assert RMSNormGated is not None
        #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        #linear attention duality
        self.linear_attn_duality = linear_attn_duality
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V,))

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
            x = Q @ KV # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
            x = Q @ KV # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            x = x + V_skip # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
            x = x.contiguous()

        return x


    def forward(self, u, H, W, seq_idx=None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]


        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        if self.linear_attn_duality:
            y = self.non_casual_linear_attn(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W
            )
        else:
            if self.kwargs.get('bidirection', False):
                #assert self.ngroups == 2 #only support bidirectional with 2 groups
                x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
                B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                dt = dt.chunk(2, dim=-1) # (B, L, nheads) -> (B, L, nheads//2)*2
                A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1) # (nheads) -> (nheads//2)*2
                y_forward = mamba_chunk_scan_combined(
                    x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    x[1].flip(1), dt[1].flip(1), A[1], B[1].flip(1), C[1].flip(1), chunk_size=self.chunk_size, D=D[1], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        y = rearrange(y, "b l h p -> b l (h p)")

        # # Multiply "gate" branch and apply extra normalization layer
        # y = self.norm(y, z)
        y = self.norm(y)
        y = y*z
        out = self.out_proj(y)
        return out


class VMAMBA2Block(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state = 64, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if kwargs.get('attn_type', 'mamba2') == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('attn_type', 'mamba2') == 'mamba2':
            self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        B, L, C = x.shape
        if H is None or W is None:
            H, W = self.input_resolution

        # --- 保证 cpe1 只在图像 token 上用 ---
        img_tokens, extra_tokens = x[:, :H*W, :], x[:, H*W:, :]   # 拆分

        img_tokens = img_tokens + self.cpe1(
            img_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        ).flatten(2).permute(0, 2, 1)

        # 再拼接回来
        x = torch.cat([img_tokens, extra_tokens], dim=1)

        # --------------------------
        shortcut = x
        x = self.norm1(x)

        # Attention可以看到额外token
        x = self.attn(x, H, W)

        x = shortcut + self.drop_path(x)

        # --- 同理，cpe2 只作用在图像token ---
        img_tokens, extra_tokens = x[:, :H*W, :], x[:, H*W:, :]

        img_tokens = img_tokens + self.cpe2(
        img_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        ).flatten(2).permute(0, 2, 1)

        x = torch.cat([img_tokens, extra_tokens], dim=1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, linear_attn_duality=False, d_state=64, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VMAMBA2Block(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, ssd_chunk_size=ssd_chunk_size,
                      linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H=None, W=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class GeoMFVSSD(nn.Module):
    """GeoMF with the VSSD visual backbone."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=60,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality= True, d_state=64, num_fusion=1, 
                 use_cross_mlp=True, use_cross_mlp_num=1, use_se=True, use_film=True, use_ffn=True, 
                 use_meta=True, use_text=True, use_simple_mlp_fusion=False, loc_input_dim=24, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.simple_downsample = kwargs.get('simple_downsample', False)
        self.simple_patch_embed = kwargs.get('simple_patch_embed', False)
        self.attn_types = kwargs.get('attn_types', ['mamba2', 'mamba2', 'mamba2', 'standard'])
        if self.simple_patch_embed:
            self.patch_embed = SimpleStem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if self.simple_downsample:
            PatchMergingBlock = SimplePatchMerging
        else:
            PatchMergingBlock = PatchMerging
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # loc_input_dim 用于 RegionPE 消融实验，默认为 24
        self.loc = nn.Sequential(
            nn.Linear(loc_input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, 256),   # 输出 256
            nn.LayerNorm(256),
        )
        # Preserve module keys used in existing checkpoints.
        self.gnn = GSCM(
            hidden_dim=256,
            num_nodes=25,
            num_iters=5,
            num_classes=60,   # 输出 256
            gnn_type=TransformerConv,
            edge_index=None
        )

        drop = 0.0
        # -------- Meta ----------
        '''self.meta_project_96 = nn.Sequential(
            nn.Linear(60, 96),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(96)
        )
        self.meta_project_192 = nn.Sequential(
            nn.Linear(60, 192),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(192)
        )'''
        self.meta_project_384 = nn.Sequential(
            nn.Linear(256, 384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(384)
        )
        '''self.meta_project_768 = nn.Sequential(
            nn.Linear(256, 768),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(768)
        )'''
        
        # -------- Image ----------
        self.img_project_384 = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(384)
        )

        # -------- Text ----------
        '''self.text_project_96 = nn.Sequential(
            nn.Linear(60, 96),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(96)
        )
        self.text_project_192 = nn.Sequential(
            nn.Linear(60, 192),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(192)
        )'''
        self.text_project_384 = nn.Sequential(
            nn.Linear(256, 384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(384)
        )
        '''self.text_project_768 = nn.Sequential(
            nn.Linear(256, 768),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(768)
        )'''
        
        self.num_fusion = num_fusion
        self.use_cross_mlp = use_cross_mlp
        self.use_cross_mlp_num = use_cross_mlp_num
        self.use_se = use_se
        self.use_film = use_film
        self.use_ffn = use_ffn
        self.use_meta = use_meta
        self.use_text = use_text
        self.use_simple_mlp_fusion = use_simple_mlp_fusion

        self.late_fusion_blocks = nn.ModuleList([
            HAFM(
                dim=384,
                dropout=0.0,
                ff_mult=4,
                use_cross_mlp=self.use_cross_mlp,
                use_cross_mlp_num=self.use_cross_mlp_num,
                use_se=self.use_se,
                use_film=self.use_film,
                use_ffn=self.use_ffn,
                use_meta=self.use_meta,
                use_text=self.use_text,
            )
            for _ in range(self.num_fusion)
        ])
        self.mlp_fusion = nn.Sequential(
            nn.Linear(3 * 384, 384),
            nn.GELU(),
            nn.LayerNorm(384)
        )
        self.head_60 = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(384),
            nn.Linear(384, 60)
        )

        self.head_10000 = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(384),
            nn.Linear(384, 10000)
        )
        #对比算法模块定义
        #Concat
        '''self.linear_fuse = nn.Linear(2 * 384, 384)'''
        
        #Gascnet
        '''self.mu_param = nn.Parameter(torch.tensor(0.0))'''

        #Dynamic
        '''self.loc_att = get_dynamic_mlp(384, SimpleNamespace(
            mlp_d=32,
            mlp_h=512,
            mlp_n=2,
            mlp_type='c'
        ))'''
        #priors
        '''self.head_meta = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 60)
        )'''
        #SAMLP
        '''self.samlp = LateFusionClassifier(img_feat_dim=384)'''

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs['attn_type'] = self.attn_types[i_layer]
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               ssd_chunk_size=ssd_chunk_size,
                               linear_attn_duality = linear_attn_duality,
                               d_state = d_state,
                               **kwargs)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.no_grad()
    def flops_and_params(model,
                     img_shape=(3, 224, 224),
                     meta_dim=128,
                     text_dim=256,
                     device="cuda",
                     use_ptflops=False,
                     log_dir="./",
                     verbose=True):
        """
        计算多模态模型参数量 (Params) 和 FLOPs
        - fvcore：支持复杂模型（推荐）
        - ptflops：对 CNN/Transformer 更精确，但只支持单模态
        """

        model = model.eval().to(device)

        # -------- 参数量 --------
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # -------- FLOPs --------
        flops = 0
        warning_lines = []

        if not use_ptflops:
            try:
                # 包一层 Wrapper，给模型自动加 dummy meta/text 输入
                class Wrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, x):
                        B = x.size(0)
                        dummy_meta_inner = torch.randn(B, meta_dim, device=x.device)
                        dummy_text_inner = torch.randn(B, text_dim, device=x.device)
                        return self.model(x, dummy_text_inner, dummy_meta_inner)

                dummy_img = torch.randn((1, *img_shape), device=device)
                wrapped_model = Wrapper(model).eval().to(device)

                flops_analyzer = FlopCountAnalysis(wrapped_model, dummy_img)
                flops = flops_analyzer.total()

                # 打印 per-op 统计，方便定位算力消耗
                if verbose:
                    print(flops_analyzer)

                unsupported_ops = flops_analyzer.unsupported_ops()
                if unsupported_ops:
                    warning_lines.append(f"[Warning] {len(unsupported_ops)} unsupported ops ignored:")
                    for k, v in unsupported_ops.items():
                        warning_lines.append(f"  - {k}: {v} time(s)")

            except Exception as e:
                if verbose:
                    print(f"[Error] fvcore FLOPs 计算异常: {e}, 返回 approximate FLOPs 1e9")
                flops = 1e9

        else:
            # 用 ptflops 更精确（对 CNN + Transformer 支持更好，但只支持单模态输入）
            with torch.cuda.device(0):
                macs, params_str = get_model_complexity_info(
                    model,
                    (3, img_shape[1], img_shape[2]),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False
                )
                flops = macs  # MACs≈FLOPs
                params = params_str

        # -------- 输出 --------
        if verbose:
            print(f"Params: {params/1e6:.2f} M")
            print(f"FLOPs: {flops/1e9:.2f} G (approx)")

        # -------- 保存日志 --------
        if warning_lines:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "flops_log.txt")
            with open(log_path, "w") as f:
                f.write("\n".join(warning_lines))
            if verbose:
                print(f"[Info] Unsupported ops details saved to {log_path}")

        return params, flops

    def data_fusion(self, img, metadata, f):
        self.f = f
        if self.f == 1: #ConcatNet

            fusion = torch.cat([img, metadata], dim=-1)  # [B, 2*D]
            fusion = self.linear_fuse(fusion)  # [B, D]

        if self.f == 3: #GeoNet

            fusion = img + metadata  # [B, D]

        
        if self.f == 4: #gasc-net

            mu_prime = torch.sigmoid(self.mu_param)

            # 按 μ' 融合图像与元特征
            mix_feat = torch.cat([
                mu_prime * img,
                (1 - mu_prime) * metadata
            ], dim=-1)  # [B, 2*C]


            weight = torch.sigmoid(self.linear_fuse(mix_feat))  # [B, D]
            fusion = weight * img 

        if self.f == 5:  # Dynamic MLP

            fusion = self.loc_att(img, metadata)
        
        if self.f == 6:  # Samlp
            """
            img: [B, N, 768]
            metadata: [B, D]  # 原始元数据向量
            """
            fusion = self.samlp(img, metadata)
        return fusion
        
    def forward_features(self, x, metadata, text):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        H, W = H//4, W//4 # downsampled by patch_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, H, W)
            H, W = H//2, W//2

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        x = self.img_project_384(x)
        metadata_2 = self.meta_project_384(metadata)  # [B, D]

        text_2 = self.text_project_384(text)  # [B, D]
        #text_2 = text
        if self.use_simple_mlp_fusion:
            ###MLP Fusion
            x = self.mlp_fusion(torch.cat([x, metadata_2, text_2], dim=-1))
        else:
            ###HAFM
            for block in self.late_fusion_blocks:
                x = block(x, metadata_2, text_2)
        
        return x

    def forward(self, x, text, metadata, return_features=False):
        metadata = self.loc(metadata)
        text = self.gnn(text)
        x = self.forward_features(x, metadata, text)
        x = self.head_60(x)
        return x


# Legacy class name for backbone imports and serialized models.
VMAMBA2 = GeoMFVSSD


class Backbone_VMAMBA2(VMAMBA2):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, **kwargs):
        super().__init__(**kwargs)
        norm_layer = nn.LayerNorm

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.head
        del self.norm
        del self.avgpool
        self.load_pretrained(pretrained,key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt} from {key}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


    def forward(self, x):

        def layer_forward(l, x, H=None, W=None):
            for blk in l.blocks:
                x = blk(x, H, W)
            if l.downsample is not None:
                y = l.downsample(x, H, W)
            else:
                y = x
            return x, y

        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        if self.simple_patch_embed:
            H, W = H//4, W//4
        else:
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, H, W)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
                outs.append(out.contiguous())
            #calculate H, W for next layer, with conv stride 3, stride 2 and padding 1
            H, W = int((H-1)/2)+1, int((W-1)/2)+1

        if len(self.out_indices) == 0:
            return x

        return outs
