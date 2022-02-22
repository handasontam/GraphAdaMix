import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
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
        super(MLP, self).__init__()
        self.num_gcns = num_gcns
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.device = device

        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for gcn_id in range(num_gcns):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        print(f"{self.num_parameters_per_gcns=}")

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        y_hats = []
        org_x = x
        for gcn_id in range(self.num_gcns):
            x = org_x
            for layer_id in range(self.num_layers - 1):
                x = self.lins[gcn_id * self.num_layers + layer_id](x)
                x = self.bns[gcn_id * (self.num_layers - 1) + layer_id](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[gcn_id * self.num_layers + self.num_layers - 1](x)
            y_hats.append(x)
        return y_hats  # list of [N * C] of size K

    def get_train_outs(self, x, edge_index, edge_attr, train_idx):
        outs_train = self.forward(x, edge_index, edge_attr)
        outs_train = [
            out_train[train_idx] for out_train in outs_train
        ]  # list of [N_train * C] of size k
        return outs_train

    @torch.no_grad()
    def inference(self, x, edge_index, edge_attr):
        self.eval()
        outs = self.forward(x, edge_index, edge_attr)  # List of [N*C] of size K
        outs = [out.softmax(dim=-1) for out in outs]  # List of [N*C] of size K
        outs = torch.stack(outs)  # K*N*C
        outs = outs.permute(1, 0, 2)  # N*K*C
        return outs


def get_optimizer(num_gcns, model, lr, mixture_embeds):
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": mixture_embeds.parameters()}],
        lr=lr,
    )
    return optimizer
