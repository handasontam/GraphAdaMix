import torch
from torch_geometric.nn import GCNConv
from models.normalize import Normalize
import copy

# from misc.gram import MixtureEmbeddings
from torch.nn import Embedding


class GConv(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_hidden_dim,
        embedding_dim,
        num_layers,
        num_gcns,
        num_nodes,
        encoder_norm="batch",
        device="cuda",
    ):
        super(GConv, self).__init__()
        self.num_gcns = num_gcns
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.activation = torch.nn.PReLU()

        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.device = device

        for gcn_id in range(num_gcns):
            self.layers.append(GCNConv(input_dim, encoder_hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(encoder_hidden_dim, encoder_hidden_dim))
            self.layers.append(GCNConv(encoder_hidden_dim, embedding_dim))

            for _ in range(num_layers - 1):
                self.bns.append(Normalize(encoder_hidden_dim, norm=encoder_norm))
            self.bns.append(Normalize(embedding_dim, norm=encoder_norm))

        self.mixture_embeds = Embedding(num_nodes, num_gcns)  # N*K
        self.mixture_embeds.weight.data = (
            1 / num_gcns * torch.ones(num_nodes, num_gcns).to(device)
        )

    def reset_parameters(self):
        self.mixture_embeds.weight.data = (
            1 / self.num_gcns * torch.zeros(self.num_nodes, self.num_gcns).to(device)
        )
        for conv in self.num_gcns:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.shape[0]
        ## mixture_dist = self.mixture_embeds.weight.softmax(dim=-1)  # N * K
        out_z = torch.zeros((num_nodes, self.embedding_dim)).to(self.device)
        data_x = x
        mixture_dist = self.mixture_embeds.weight.softmax(dim=-1)
        for gcn_id in range(self.num_gcns):
            z = data_x
            for layer_id in range(self.num_layers):
                z = self.layers[gcn_id * self.num_layers + layer_id](
                    z, edge_index, edge_weight
                )
                z = self.bns[gcn_id * self.num_layers + layer_id](z)
                z = self.activation(z)
            # z is N * embedding_dim
            # out_z += self.mixture_embeds.weight[:, gcn_id].unsqueeze(-1) * z
            out_z += mixture_dist[:, gcn_id].unsqueeze(-1) * z
        return out_z  # N * embedding_dim

    def forward_k_embed(self, x, edge_index, edge_weight=None):
        outs = []
        num_nodes = x.shape[0]
        ## mixture_dist = self.mixture_embeds.weight.softmax(dim=-1)  # N * K
        data_x = x
        for gcn_id in range(self.num_gcns):
            z = data_x
            for layer_id in range(self.num_layers):
                z = self.layers[gcn_id * self.num_layers + layer_id](
                    z, edge_index, edge_weight
                )
                z = self.bns[gcn_id * self.num_layers + layer_id](z)
                z = self.activation(z)
            # z is N * embedding_dim
            outs.append(z)
        return outs  #  List of [N*embedding_dim] of size K


class Encoder(torch.nn.Module):
    def __init__(
        self, encoder, augmentor, embedding_dim, hidden_dim, predictor_norm="batch"
    ):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(
            self.get_target_encoder().parameters(), self.online_encoder.parameters()
        ):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1 = self.online_encoder(x1, edge_index1, edge_weight1)
        h2 = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1)
        h2_pred = self.predictor(h2)

        with torch.no_grad():
            h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target
