import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from misc import gram
from tqdm import tqdm


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_gcns,
        num_nodes,
        use_bns,
        device,
    ):
        super(SAGE, self).__init__()
        self.num_gcns = num_gcns
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_classes = out_channels
        self.use_bns = use_bns
        self.device = device

        self.convs = torch.nn.ModuleList()
        if self.use_bns:
            self.bns = torch.nn.ModuleList()
        for _ in range(num_gcns):
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            if self.use_bns:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                if self.use_bns:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

        self.num_parameters_per_gcns = gram.count_parameters(self.convs) / self.num_gcns
        print(f"{self.num_parameters_per_gcns=}")

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bns:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index):
        y_hats = []
        ori_x = x
        for gcn_id in range(self.num_gcns):
            x = ori_x
            for layer_id in range(self.num_layers - 1):
                x = self.convs[gcn_id * self.num_layers + layer_id](x, edge_index)
                x = self.bns[gcn_id * (self.num_layers - 1) + layer_id](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[gcn_id * self.num_layers + self.num_layers - 1](
                x, edge_index
            )
            y_hats.append(x)
        return y_hats

    def get_train_outs(self, loader, train_idx):
        outs_train = [
            torch.zeros(self.num_nodes, self.num_classes).to(self.device)
            for _ in range(self.num_gcns)
        ]

        for batch_data in loader:
            batch_data = batch_data.to(self.device)
            outs = self.forward(
                batch_data.x, batch_data.edge_index
            )  # list of [N_batch * C] of size K
            y_hats = [
                out[batch_data.train_mask] for out in outs
            ]  # list of [N_batch_train * C] of size k
            batch_idx = batch_data.idx[batch_data.train_mask].squeeze(1)
            for out_train, y_hat in zip(outs_train, y_hats):
                out_train[batch_idx] = y_hat
        outs_train = [out_train[train_idx] for out_train in outs_train]
        return outs_train  # List of (N_train X C) of size k

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description("Evaluating")

        y_hats = []  # List of [N * C] of size k
        ori_x_all = x_all

        for gcn_id in range(self.num_gcns):
            x_all = ori_x_all
            for layer_id in range(self.num_layers):
                xs = []
                for batch_size, n_id, adj in subgraph_loader:
                    edge_index, _, size = adj.to(device)
                    x = x_all[n_id].to(device)
                    x_target = x[: size[1]]
                    x = self.convs[gcn_id * self.num_layers + layer_id](
                        (x, x_target), edge_index
                    )
                    if layer_id != self.num_layers - 1:
                        x = self.bns[gcn_id * (self.num_layers - 1) + layer_id](x)
                        x = F.relu(x)
                    xs.append(x)

                    pbar.update(batch_size)

                x_all = torch.cat(xs, dim=0)
            y_hats.append(x_all)
        pbar.close()
        y_hats = [out.softmax(dim=-1) for out in y_hats]  # list of (N * C) of size k
        y_hats = torch.stack(y_hats)  # K*N*C
        y_hats = y_hats.permute(1, 0, 2)  # N*K*C

        return y_hats  # N*K*C
