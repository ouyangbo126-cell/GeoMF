"""Trainable semantic and fusion modules of GeoMF."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data


class GSCM(nn.Module):
    """Graph-based semantic correlation module for numeric ship attributes."""

    def __init__(self, hidden_dim=128, num_nodes=25, num_iters=5, num_classes=10, gnn_type='gcn', edge_index=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters
        self.learnable_nodes = nn.Parameter(torch.randn(6, 1))

        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU()
        )

        # 传入的 gnn_type 直接就是类了，比如 TransformerConv
        GNNLayer = gnn_type
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_iters)
        ])


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        if edge_index is None:
            # 默认图结构
            self.register_buffer('edge_index', torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                [19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 22, 23, 23, 15, 23, 17, 24, 24, 24, 24, 24]
            ], dtype=torch.long))
        else:
            if edge_index.shape[0] != 2:
                raise ValueError("edge_index must be shape [2, num_edges]")
            self.register_buffer('edge_index', edge_index)

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(-1)  # (B, num_nodes) -> (B, num_nodes, 1)
        learnable_nodes = self.learnable_nodes.unsqueeze(0).expand(B, -1, -1)  # (B, 6, 1)
        x_embed = torch.cat([x, learnable_nodes], dim=1)  # (B, num_nodes+6, 1)
        h = self.input_proj(x_embed)  # (B, num_nodes+6, hidden_dim)

        data_list = [Data(x=h[i], edge_index=self.edge_index) for i in range(B)]
        batch = Batch.from_data_list(data_list)

        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
            x = F.relu(x)

        root_indices = torch.arange(24, x.size(0), step=25, device=x.device)
        graph_feature = x[root_indices]

        return graph_feature



class HAFM(nn.Module):
    """Hierarchical adaptive fusion of visual, geographic, and semantic features."""

    def __init__(self, dim, dropout=0.0, ff_mult=4,
                use_cross_mlp=True, 
                use_cross_mlp_num=3, 
                use_se=False, 
                use_film=True, 
                use_ffn=True,
                use_meta=True,
                use_text=True):
        super().__init__()
        self.dim = dim

        # 保存消融参数
        self.use_cross_mlp = use_cross_mlp
        self.use_cross_mlp_num = use_cross_mlp_num
        self.use_se = use_se
        self.use_film = use_film
        self.use_ffn = use_ffn
        self.use_meta = use_meta
        self.use_text = use_text

        # 输入 LayerNorm
        self.input_ln = nn.LayerNorm(dim)

        # -------- Cross MLP --------
        if use_cross_mlp:
            in_dim = 2 * dim
            self.cross_layers = nn.ModuleList()
            for i in range(use_cross_mlp_num):
                self.cross_layers.append(nn.Sequential(
                    nn.Linear(in_dim, 4 * dim),
                    nn.GELU(),
                    nn.Linear(4 * dim, in_dim),
                    nn.LayerNorm(in_dim)
                ))
            self.cross_project = nn.Linear(in_dim, dim)
            self.alpha_cross = nn.Parameter(torch.tensor(0.0))


        # -------- Dynamic SE --------
        if use_se:
            self.se_fc1 = nn.Linear(3*dim, dim//4)
            self.se_fc2 = nn.Linear(dim//4, dim)

        # -------- FiLM --------
        if use_film:
            self.film_scale = nn.Linear(2*dim, dim)
            self.film_shift = nn.Linear(2*dim, dim)
            self.film_weight = nn.Parameter(torch.tensor(0.0))

        # -------- FFN + Gate --------
        if use_ffn:
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ff_mult*dim),
                nn.GELU(),
                nn.Linear(ff_mult*dim, dim),
                nn.Dropout(dropout)
            )
            self.ffn_gate = nn.Linear(2*dim, dim)
            self.gate_weight = nn.Parameter(torch.tensor(0.0))
            self.cond_proj = nn.Linear(2*dim, dim)

        # -------- Bias Projection（始终存在）--------
        self.aux_proj = nn.Linear(dim, dim)
        self.out_ln = nn.LayerNorm(dim)

        # 打印初始化参数
        print("[HAFM] init with params:")
        print(f"  use_cross_mlp = {self.use_cross_mlp}")
        print(f"  use_cross_mlp_num = {self.use_cross_mlp_num}")
        print(f"  use_se        = {self.use_se}")
        print(f"  use_film      = {self.use_film}")
        print(f"  use_ffn       = {self.use_ffn}")
        print(f"  use_meta       = {self.use_meta}")
        print(f"  use_text       = {self.use_text}")

    def _apply_acfu(self, fusion, aux):
        # -------- Cross MLP 残差 --------
        if self.use_cross_mlp:
            fusion_cat = torch.cat([fusion, aux], dim=-1)
            for layer in self.cross_layers:
                fusion_cat = fusion_cat + layer(fusion_cat)  # 每层残差
            fusion_proj = self.cross_project(fusion_cat)
            alpha = torch.sigmoid(self.alpha_cross)
            fusion = (1 - alpha) * fusion + alpha * aux + fusion_proj


        # -------- Dynamic SE --------
        if self.use_se and hasattr(self, "se_fc1"):
            se_input = torch.cat([fusion, aux, aux], dim=-1)
            se = F.relu(self.se_fc1(se_input))
            se = torch.sigmoid(self.se_fc2(se))
            fusion = fusion * se

        # -------- FiLM --------
        if self.use_film and hasattr(self, "film_scale"):
            cond = torch.cat([aux, aux], dim=-1)
            scale = self.film_scale(cond)
            shift = self.film_shift(cond)
            fusion = fusion * (1 + torch.sigmoid(self.film_weight) * scale) \
                    + torch.sigmoid(self.film_weight) * shift

        # -------- FFN + Gate --------
        if self.use_ffn and hasattr(self, "ffn"):
            cond_proj = self.cond_proj(torch.cat([aux, aux], dim=-1))
            gate = torch.sigmoid(self.ffn_gate(torch.cat([fusion, cond_proj], dim=-1)))
            fusion = fusion + torch.sigmoid(self.gate_weight) * gate * self.ffn(self.ffn_ln(fusion))

        return fusion

    cross_attn_block = _apply_acfu  # Legacy method name

    def forward(self, img, metadata, text):
        fusion = self.input_ln(img)

        # 共享 cross-attention
        if self.use_meta == True:
            fusion = self._apply_acfu(fusion, metadata)
        if self.use_text == True:
            fusion = self._apply_acfu(fusion, text)

        # Bias 投影
        if self.use_meta == True:
            fusion = fusion + self.aux_proj(metadata) 
        if self.use_text == True:
            fusion = fusion + self.aux_proj(text)

        return self.out_ln(fusion)
