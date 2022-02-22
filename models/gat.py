import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # noqa
from torch.nn import Embedding


class GAT(torch.nn.Module):
    def __init__(
        self, num_gcns, num_nodes, num_features, num_hidden, num_classes, device
    ):
        super(GAT, self).__init__()
        self.num_gcns = num_gcns
        self.num_nodes = num_nodes
        self.device = device
        self.convs = torch.nn.ModuleList()

        for gcn_id in range(num_gcns):
            self.convs.append(GATConv(num_features, num_hidden, heads=8, dropout=0.6))
            self.convs.append(
                GATConv(num_hidden * 8, num_classes, heads=1, concat=False, dropout=0.6)
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        y_hats = []
        data_x = F.dropout(x, p=0.6, training=self.training)
        for gcn_id in range(self.num_gcns):
            x, edge_index, edge_weight = data_x, edge_index, edge_attr
            x = F.elu(self.convs[gcn_id * 2 + 0](x, edge_index, edge_weight))
            x = F.dropout(x, p=0.6, training=self.training)
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
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), weight_decay=5e-4),
            dict(params=mixture_embeds.parameters(), weight_decay=0),
        ],
        lr=lr,
    )  # Only perform weight-decay on first convolution.
    return optimizer
