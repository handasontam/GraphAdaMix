import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # noqa
from torch.nn import Embedding
from misc import gram


class PlanetoidGCN(torch.nn.Module):
    def __init__(
        self,
        num_gcns,
        num_nodes,
        num_features,
        num_hidden,
        num_classes,
        use_gdc,
        device,
    ):
        super(PlanetoidGCN, self).__init__()
        self.num_gcns = num_gcns
        self.num_nodes = num_nodes
        self.device = device
        self.convs = torch.nn.ModuleList()

        for gcn_id in range(num_gcns):
            self.convs.append(GCNConv(num_features, num_hidden, normalize=not use_gdc))
            self.convs.append(GCNConv(num_hidden, num_classes, normalize=not use_gdc))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        y_hats = []
        data_x = x
        for gcn_id in range(self.num_gcns):
            x, edge_index, edge_weight = data_x, edge_index, edge_attr
            x = F.relu(self.convs[gcn_id * 2 + 0](x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = self.convs[gcn_id * 2 + 1](x, edge_index, edge_weight)
            y_hats.append(x)
        return y_hats

    def get_train_outs(self, x, edge_index, edge_attr, train_idx):
        outs_train = self.forward(x, edge_index, edge_attr)
        outs_train = [
            out_train[train_idx] for out_train in outs_train
        ]  # list of [N_train * C] of size k
        return outs_train

    @torch.no_grad()
    def inference(self, x, edge_index, edge_attr):
        outs = self.forward(x, edge_index, edge_attr)  # list of (N * C) of size k
        outs = [out.softmax(dim=-1) for out in outs]  # list of (N * C) of size k
        outs = torch.stack(outs)  # K*N*C
        outs = outs.permute(1, 0, 2)  # N*K*C
        return outs


def get_optimizer(num_gcns, model, lr, mixture_embeds):
    params = []
    for gcn_id in range(num_gcns):
        params.extend(
            [
                dict(
                    params=model.convs[gcn_id * 2 + 0].parameters(), weight_decay=5e-4
                ),
                dict(params=model.convs[gcn_id * 2 + 1].parameters(), weight_decay=0),
            ]
        )
    params.extend([dict(params=mixture_embeds.parameters(), weight_decay=0)])
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
    )  # Only perform weight-decay on first convolution.
    return optimizer


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_gcns,
        num_nodes,
        device,
    ):
        super(GCN, self).__init__()
        self.num_gcns = num_gcns
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_classes = out_channels
        self.device = device

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_gcns):
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

        self.num_parameters_per_gcns = gram.count_parameters(self.convs) / self.num_gcns
        print(f"{self.num_parameters_per_gcns=}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        y_hats = []
        ori_x = x
        for gcn_id in range(self.num_gcns):
            x = ori_x
            for layer_id in range(self.num_layers - 1):
                x = self.convs[gcn_id * self.num_layers + layer_id](
                    x, edge_index, edge_attr
                )
                x = self.bns[gcn_id * (self.num_layers - 1) + layer_id](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[gcn_id * self.num_layers + self.num_layers - 1](
                x, edge_index, edge_attr
            )
            y_hats.append(x)
        return y_hats

    def get_train_outs(self, x, edge_index, edge_attr, train_idx):
        outs_train = self.forward(x, edge_index, edge_attr)
        outs_train = [
            out_train[train_idx] for out_train in outs_train
        ]  # list of [N_train * C] of size k
        return outs_train

    @torch.no_grad()
    def inference(self, x, edge_index, edge_attr):
        outs = self.forward(x, edge_index, edge_attr)  # list of (N * C) of size k
        outs = [out.softmax(dim=-1) for out in outs]  # list of (N * C) of size k
        outs = torch.stack(outs)  # K*N*C
        outs = outs.permute(1, 0, 2)  # N*K*C
        return outs


def get_generic_optimizer(num_gcns, model, lr, mixture_embeds):
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": mixture_embeds.parameters()}],
        lr=lr,
    )
    return optimizer
