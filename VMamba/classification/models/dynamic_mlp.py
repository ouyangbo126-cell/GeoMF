import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from types import SimpleNamespace
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GATConv, GCNConv, GraphConv, SAGEConv, ChebConv, TAGConv, SGConv, GINConv,
    APPNP, ARMAConv, GMMConv, MFConv, GatedGraphConv, AGNNConv, DNAConv,
    TransformerConv, PointTransformerConv, NNConv, PANConv, EdgeConv, ASAPooling
)
import torch.nn.functional as F

class FCResLayer(nn.Module):
    def __init__(self, linear_size=256):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes=1000, num_filts=256):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, num_filts),
            nn.ReLU(inplace=True),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
        )
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)

    def forward(self, x, return_feats=True, class_of_interest=None):
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb  # [b, num_filts]
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return class_pred  # [b, num_class]

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])

class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(conv, )
        if not bias:
            self.conv.add_module('ln', nn.LayerNorm(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class Dynamic_MLP_C(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        cat_fea = torch.cat([img_fea, loc_fea], 1)

        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(cat_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea


class RecursiveBlock(nn.Module):
    def __init__(self, inplanes, planes, loc_planes, mlp_type='c'):
        super().__init__()
        MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def forward(self, img_fea, loc_fea):
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea


class FusionModule(nn.Module):
    def __init__(self, inplanes=768, planes=768, hidden=512, num_layers=2, mlp_type='c'):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)

        self.conv3 = nn.Linear(planes, inplanes)
        self.norm3 = nn.LayerNorm(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        '''
        img_fea: (N, channel), backbone输出经过全局池化的feature
        loc_fea: (N, fea_dim)
        '''
        identity = img_fea

        img_fea = self.conv1(img_fea)

        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        img_fea += identity

        return img_fea


def get_dynamic_mlp(inplanes, args):
    return FusionModule(inplanes=inplanes,
                        planes=args.mlp_d,
                        hidden=args.mlp_h,
                        num_layers=args.mlp_n,
                        mlp_type=args.mlp_type)

class SAMLPUnit(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(SAMLPUnit, self).__init__()
        # 多头自注意力（可按需调整头数、替换实现）
        self.self_attention = nn.MultiheadAttention(dim, num_heads=4)  
        # MLP 层（特征变换）
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        # 前馈网络（强化特征）
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x.unsqueeze(0)  # 变成 [1, batch_size, dim]

        attn_output, _ = self.self_attention(x, x, x)  # 输出形状也是 [1, batch_size, dim]

        attn_out = attn_output + x  # 残差连接，形状一样

        attn_out = attn_out.squeeze(0)  # 去掉 seq_len 维度，变回 [batch_size, dim]

        # MLP + 残差
        mlp_out = self.mlp(attn_out)
        mlp_res = mlp_out + attn_out  

        # 前馈网络 + 残差
        ff_out = self.feed_forward(mlp_res)
        final_out = ff_out + mlp_res  

        return final_out

# 封装重复4次的 SAMLP 模块
class SAMLP(nn.Module):
    def __init__(self, dim, hidden_dim, repeat_times=4):
        super(SAMLP, self).__init__()
        self.repeat_times = repeat_times
        # 构建多个 SAMLP 单元（这里复用同一个单元，也可单独初始化）
        self.samlp_units = nn.ModuleList([
            SAMLPUnit(dim, hidden_dim) for _ in range(repeat_times)
        ])
        self.head = nn.Linear(dim, 60)

    def forward(self, x):
        for unit in self.samlp_units:
            x = unit(x)  # 依次经过4次 SAMLP 单元处理
        #x = self.head(x)
        return x

class DynamicFusionBlock(nn.Module):
    def __init__(self, dim=384, hidden_dim=512):
        super().__init__()
        # 注意：使用你原本的 get_dynamic_mlp 动态注意力结构
        self.loc_att = get_dynamic_mlp(dim, SimpleNamespace(
            mlp_d=32,
            mlp_h=hidden_dim,
            mlp_n=2,
            mlp_type='c'
        ))

        # 对融合后的图像特征做 MLP
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(dim * 2, dim)
        )

        # 对 loc_fea_proj 也加一个 MLP
        self.loc_mlp = nn.Sequential(
            nn.Linear(32, 32 * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(32 * 2, 32 * 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(32 * 2, 32)
        )

    def forward(self, img_feat, loc_feat):
        attn_output = self.loc_att(img_feat, loc_feat)  # 融合，shape: (B, D)

        # 图像与loc特征分别进MLP
        img_feat = self.img_mlp(attn_output)
        loc_feat = self.loc_mlp(loc_feat)

        return img_feat, loc_feat

class LateFusionClassifier(nn.Module):
    def __init__(self, img_feat_dim=384, n_rounds=3):
        super().__init__()
        # 多模态（经纬度）处理
        #self.samlp = SAMLP(dim=24, hidden_dim=128, repeat_times=4)
        
        # 后期融合模块，多轮动态注意力融合
        self.fusion_blocks = nn.ModuleList([
            DynamicFusionBlock(dim=img_feat_dim) for _ in range(n_rounds)
        ])


    def forward(self, img_feat, metadata):
        """
        img_feat: (B, D) 图像特征，B=batch size，D=img_feat_dim
        metadata:  (B, 24) 经纬度等元信息
        """
        B = img_feat.size(0)

        # 1. loc 特征提取 + 映射
        #loc_feat = self.samlp(metadata)  # (B, 24)
        loc_feat = metadata
        # 2. 多轮融合
        for block in self.fusion_blocks:
            img_feat, loc_feat = block(img_feat, loc_feat)

        fusion_feat = img_feat

        return fusion_feat 


class CrossModalFusionBlock(nn.Module):
    def __init__(self, dim, cross_attn_heads=2, dropout=0.0, ff_mult=6):
        super().__init__()
        self.dim = dim

        # ------------------------
        # 输入归一化
        self.input_ln = nn.LayerNorm(dim)

        # ------------------------
        # Cross-Attention layers
        self.meta_cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=cross_attn_heads, batch_first=True)
            for _ in range(2)
        ])
        self.text_cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=cross_attn_heads, batch_first=True)
            for _ in range(2)
        ])
        # alpha 控制融合强弱
        self.alpha_meta_cross = nn.Parameter(torch.tensor(0.0))    # metadata 强
        self.alpha_text_cross = nn.Parameter(torch.tensor(-1.0))   # text 弱

        # ------------------------
        # FiLM 条件调制
        self.film_scale = nn.Linear(2*dim, dim)
        self.film_shift = nn.Linear(2*dim, dim)
        self.film_weight = nn.Parameter(torch.tensor(0.0))

        # ------------------------
        # 动态 SE
        self.se_fc1 = nn.Linear(3*dim, dim//2)
        self.se_fc2 = nn.Linear(dim//2, dim)

        # ------------------------
        # 条件 FFN
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_mult*dim),
            nn.GELU(),
            nn.Linear(ff_mult*dim, dim),
            nn.Dropout(dropout)
        )
        self.ffn_gate = nn.Linear(2*dim, dim)
        self.gate_weight = nn.Parameter(torch.tensor(0.0))

        # ------------------------
        # Bias 投影
        self.meta_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1,1,dim))

        # 输出归一化
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, img, metadata, text):
        B, N, D = img.shape
        fusion = self.input_ln(img)

        # ------------------------
        # Cross-Attention: metadata (主辅助信号)
        meta_exp = metadata.unsqueeze(1).expand(-1, N, -1)
        query = fusion
        key = value = meta_exp
        for attn in self.meta_cross_attn_layers:
            residual = query
            out, _ = attn(query, key, value)
            query = residual + out
        alpha_meta = torch.sigmoid(self.alpha_meta_cross)
        fusion = (1-alpha_meta)*query + alpha_meta*fusion

        # ------------------------
        # Cross-Attention: text (弱辅助信号)
        text_exp = text.unsqueeze(1).expand(-1, N, -1)
        query = fusion
        key = value = text_exp
        for attn in self.text_cross_attn_layers:
            residual = query
            out, _ = attn(query, key, value)
            query = residual + out
        alpha_text = torch.sigmoid(self.alpha_text_cross)
        fusion = (1-alpha_text)*query + alpha_text*fusion

        # ------------------------
        # 动态 SE
        fusion_mean = fusion.mean(dim=1)                        # [B,D]
        se_input = torch.cat([fusion_mean, metadata, text], dim=-1)  # [B,3D]
        se = F.relu(self.se_fc1(se_input))
        se = torch.sigmoid(self.se_fc2(se)).unsqueeze(1)       # [B,1,D]
        fusion = fusion * se

        # ------------------------
        # FiLM 条件调制
        cond = torch.cat([metadata, text], dim=-1)             # [B,2D]
        scale = self.film_scale(cond).unsqueeze(1)
        shift = self.film_shift(cond).unsqueeze(1)
        fusion = fusion * (1 + torch.sigmoid(self.film_weight)*scale) \
                 + torch.sigmoid(self.film_weight)*shift

        # ------------------------
        # 条件 FFN + gate
        attn_weights = torch.softmax(
            torch.matmul(fusion, self.pool_query.transpose(-1,-2)) / (D**0.5), dim=1
        )
        fusion_summary = (fusion * attn_weights).sum(dim=1)   # [B,D]

        # cond 投影到 dim 保证维度一致
        cond_proj = nn.Linear(cond.shape[-1], D).to(fusion.device)(cond)
        gate = torch.sigmoid(self.ffn_gate(torch.cat([fusion_summary, cond_proj], dim=-1))).unsqueeze(1)
        fusion = fusion + torch.sigmoid(self.gate_weight)*gate*self.ffn(self.ffn_ln(fusion))

        # ------------------------
        # Bias 注入
        fusion = fusion + self.meta_proj(metadata).unsqueeze(1)
        fusion = fusion + self.text_proj(text).unsqueeze(1)

        # 输出归一化
        fusion = self.out_ln(fusion)
        return fusion


# Legacy imports for existing callers and serialized models.
from .geomf_modules import GSCM as HierarchicalGNN, HAFM as LateFusionCrossModalBlock
