import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from misc import gram


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
        self.final_lins = torch.nn.ModuleList()
        for gcn_id in range(num_gcns):
            for _ in range(num_layers + 1):
                self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.final_lins.append(
                torch.nn.Linear((num_layers + 1) * hidden_channels, out_channels)
            )

        self.dropout = dropout

        self.num_parameters_per_gcns = (
            gram.count_parameters(self.lins) + gram.count_parameters(self.final_lins)
        ) / self.num_gcns

        print(f"{self.num_parameters_per_gcns=}")

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for final_lin in self.final_lins:
            final_lin.reset_parameters()

    def forward(self, sign_xs):
        y_hats = []
        for gcn_id in range(self.num_gcns):
            outs = []
            for layer_id, x in enumerate(sign_xs):
                out = F.dropout(
                    F.relu(self.lins[gcn_id * (self.num_layers + 1) + layer_id](x)),
                    p=0.5,
                    training=self.training,
                )
                outs.append(out)
            x = torch.cat(outs, dim=-1)
            x = self.final_lins[gcn_id](x)
            y_hats.append(x)
        return y_hats

    def get_train_outs(self, sign_xs):
        return self.forward(sign_xs)  # List of (N_train X C) of size k

    def inference(self, xs):
        self.eval()
        y_preds = []
        loader = DataLoader(range(self.num_nodes), batch_size=400000)
        for perm in loader:
            y_pred = self.forward([x[perm] for x in xs])  # list of (B X C) of size K
            y_pred = torch.stack(y_pred).permute(1, 0, 2)  # B X K X C
            y_preds.append(y_pred)
        final_y_pred = torch.cat(y_preds, dim=0)  # N X K X C
        return final_y_pred  # N X K X C