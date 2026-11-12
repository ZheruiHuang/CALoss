import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import SelfAttnBlock, AxialPairAttnBlock, GatedGraphNodeEncoder


REL_FEATURE_DIM = 5


class PairPredictor(nn.Module):
    def __init__(self, hid_dim, n_heads, attn_layers, axial_attn_layers, dropout):
        super().__init__()
        self.node_attn_layers = nn.ModuleList(
            [SelfAttnBlock(hid_dim, n_heads, dropout) for _ in range(attn_layers)]
        )
        self.rel2attn = nn.Sequential(nn.Linear(3, n_heads), nn.Tanh())

        self.origin_proj = nn.Linear(hid_dim, hid_dim)
        self.dest_proj = nn.Linear(hid_dim, hid_dim)
        self.rel_mlp = nn.Sequential(nn.Linear(REL_FEATURE_DIM, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))

        self.lin_o   = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_d   = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_od  = nn.Linear(hid_dim, hid_dim, bias=False)
        self.lin_rel = nn.Linear(hid_dim, hid_dim, bias=False)

        self.pair_axial_layers = nn.ModuleList(
            [AxialPairAttnBlock(hid_dim, n_heads, dropout) for _ in range(axial_attn_layers)]
        )

        self.pair2node = nn.Sequential(
            nn.LayerNorm(hid_dim * 2),
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, h0, cents, ap):
        N = h0.size(0)

        delta = cents.unsqueeze(1) - cents.unsqueeze(0)
        dist = torch.norm(delta, dim=-1, keepdim=True)
        attn_bias = self.rel2attn(torch.cat([delta, dist], -1)).permute(2, 0, 1).contiguous()

        aff_ff  = ap["aff_ff"].unsqueeze(-1)
        aff_hop = ap["aff_hop"].unsqueeze(-1)
        rel_emb = self.rel_mlp(torch.cat([delta, dist, aff_ff, aff_hop], dim=-1))

        o0 = self.origin_proj(h0)
        d0 = self.dest_proj(h0)
        o0e = o0.unsqueeze(1).expand(-1, N, -1)
        d0e = d0.unsqueeze(0).expand(N, -1, -1)
        od0 = o0e * d0e

        pair_emb = (
            self.lin_o(o0e) +
            self.lin_d(d0e) +
            self.lin_od(od0) +
            self.lin_rel(rel_emb)
        )

        for layer in self.pair_axial_layers:
            pair_emb = layer(pair_emb)

        h_o = pair_emb.mean(dim=1)
        h_d = pair_emb.mean(dim=0)
        h_aggr = self.pair2node(torch.cat([h_o, h_d], dim=-1))

        h = h0 + h_aggr
        for layer in self.node_attn_layers:
            h = layer(h, attn_bias=attn_bias)

        o = self.origin_proj(h)
        d = self.dest_proj(h)
        oe = o.unsqueeze(1).expand(-1, N, -1)
        de = d.unsqueeze(0).expand(N, -1, -1)
        od = oe * de

        final_pair_feat = torch.cat([oe, de, od], dim=-1)
        pred_od = self.out_mlp(final_pair_feat).squeeze(-1)

        return pred_od


class PredODGModel(nn.Module):
    def __init__(self, feat_dim, hid_dim, encode_mlp_layers,
                 encode_gattn_layers, attn_heads, attn_layers, axial_attn_layers,
                 dropout):
        super().__init__()
        self.encoder = GatedGraphNodeEncoder(
            feat_dim, hid_dim, hid_dim, encode_mlp_layers, encode_gattn_layers
        )
        self.predictor = PairPredictor(hid_dim, attn_heads, attn_layers,
                                       axial_attn_layers, dropout)

    def forward(self, feat_lst, centroid_lst, adj_pack_lst):
        """
        feat_lst: list of B features, each of shape (N_i, F)
        return: list of predicted OD matrices, each of shape (N_i, N_i)
        """
        node_emb_lst = self.encoder(feat_lst, centroid_lst, adj_pack_lst)

        pred_od_lst = []
        for h, cents, ap in zip(node_emb_lst, centroid_lst, adj_pack_lst):
            pred_od = self.predictor(h, cents, ap)
            pred_od_lst.append(pred_od)

        return pred_od_lst
