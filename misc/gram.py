import torch
import torch.nn.functional as F
from torch.nn import Embedding
import numpy as np
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.datasets import WikiCS, Amazon, Coauthor, Planetoid
from torch_geometric.data import InMemoryDataset, Data
from typing import Optional, Callable, List
import os
import pandas as pd
from datetime import date


class MixtureEmbeddings(torch.nn.Module):
    def __init__(self, num_nodes, num_gcns, device):
        super(MixtureEmbeddings, self).__init__()
        self.num_nodes = num_nodes
        self.num_gcns = num_gcns
        self.mixture_embeds = Embedding(
            num_embeddings=num_nodes, embedding_dim=self.num_gcns
        )  # N * K
        self.mixture_embeds.weight.data = torch.zeros(num_nodes, num_gcns).to(device)

    def reset_parameters(self):
        self.mixture_embeds.weight.data = torch.zeros(self.num_nodes, self.num_gcns).to(
            self.device
        )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jsd(net_1_logits, net_2_logits):
    net_1_probs = F.softmax(net_1_logits, dim=1)
    net_2_probs = F.softmax(net_2_logits, dim=1)

    m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(m.log(), net_1_probs, reduction="batchmean")
    loss += F.kl_div(m.log(), net_2_probs, reduction="batchmean")

    return 0.5 * loss


def edge_jsd_loss(sampled_pairs, mixture_embeds):
    """
    Args:
        sampled_pairs (#Samples X 2) tensor
    """
    embeds_pair = mixture_embeds(sampled_pairs)  # (#Samples X 2 X #Classes) tensor
    P = embeds_pair[:, 0, :]  # (#Samples X #Classes)
    Q = embeds_pair[:, 1, :]  # (#Samples X #Classes)
    return jsd(P, Q)


def get_pi_loss(num_nodes, train_idx, gamma, mixture_embeds, device):
    ln_Pi = mixture_embeds(torch.arange(num_nodes).to(device)).log_softmax(
        dim=-1
    )  # N*k, ln(pi_nk)
    train_ln_Pi = ln_Pi[train_idx]  # N_Train * k
    pi_loss = -(gamma[train_idx] * train_ln_Pi).mean()
    return pi_loss


def optimize_pi(
    optimizer_pi, edge_index, num_nodes, train_mask, gamma, rho, mixture_embeds, device
):
    optimizer_pi.zero_grad()
    pi_loss = get_pi_loss(num_nodes, train_mask, gamma, mixture_embeds, device)
    jsd_loss = edge_jsd_loss(edge_index.T, mixture_embeds)

    reg_loss = pi_loss + rho * jsd_loss
    reg_loss.backward()
    optimizer_pi.step()
    return pi_loss, jsd_loss


def get_nll_loss(gamma, unnormalized_y_hats, y, mixture_embeds):
    """[compoute loss for backprop]

    Args:
        gamma (tensor): [N*K tensor]
        y_hats ([type]): List of [N_train X C] of size k
        y ([type]): [description]
    """

    # \sum_n \sum_k \gamma_{nk} (\ln \pi_{nk})
    ln_Pi = mixture_embeds.log_softmax(dim=-1)  # N_train*k , ln(pi_nk)
    pi_loss = -(gamma.detach() * ln_Pi).mean()
    # negative log-likelihood
    u_y_hats = unnormalized_y_hats  # List of [N_train X C] of size k
    nll = torch.stack(
        [
            F.nll_loss(u_y_hat.log_softmax(dim=-1), y, reduction="none")
            for u_y_hat in u_y_hats
        ],
        dim=0,
    ).T  # a N*k tensor
    loss = (gamma.detach() * nll).sum(dim=-1)  # N * 1 tensor
    loss = torch.mean(loss)
    return pi_loss + loss


def e_step_optimize(
    model,
    outs_train,  # list of (N_train * C) of size k
    y_train,
    mixture_embeds,
    train_idx,
    num_nodes,
    num_gcns,
    device,
):
    # list of nll for each model

    with torch.no_grad():
        model.eval()
        gamma = torch.ones(num_nodes, num_gcns).to(device)  # N * k
        y_hats = [
            train_out.softmax(dim=-1) for train_out in outs_train
        ]  # list of (N_train * C) of size k

        # negative likelihood (without log)
        nl = torch.stack(
            [F.nll_loss(y_hat, y_train, reduction="none") for y_hat in y_hats],
            dim=0,
        ).T  # N_train * k

        likelihood = -nl  # N_train * k

        Pi = mixture_embeds(train_idx).to(device).softmax(dim=-1)  # N_train * k

        unnormalized_gamma = Pi * likelihood  # N_train * k
        gamma[train_idx] = F.normalize(unnormalized_gamma, p=1, dim=1, eps=1e-45)
    return gamma.detach()  # N * k


def m_step_optimize(
    model,
    optimizer,
    train_outs,
    edge_index,
    train_y,
    train_gamma,
    rho,
    train_mixture_embeds,
    mixture_embeds,
    mixture_prop_sample_size,
    device,
):
    num_edges = edge_index.shape[1]

    # train_gamma = gamma[train_mask]
    # train_mixture_embeds = mixture_embeds(train_mask)
    # mixture_embeds(torch.arange(num_nodes).to(device))
    # Optimize model's parameters:
    # model_loss = optimize_model_params(
    # model, optimizer, train_gamma, train_outs, train_y, train_mixture_embeds
    # )

    model.train()
    optimizer.zero_grad()

    # NLL loss
    loss = get_nll_loss(train_gamma, train_outs, train_y, train_mixture_embeds)
    # Mixture Propagation loss
    sampled_edges = torch.tensor(
        np.random.choice(
            range(num_edges),
            size=min(num_edges, mixture_prop_sample_size),
            replace=False,
        )
    )
    jsd_loss = edge_jsd_loss(
        edge_index[:, sampled_edges].T,
        mixture_embeds,
    )

    (loss + rho * jsd_loss).backward()
    optimizer.step()

    return loss.item(), jsd_loss.item()


def predict(softmax_outs, mixture_embeds, device):
    # out = [out.softmax(dim=-1) for out in outs]
    # out = torch.stack(out)  # K*N*C
    # outs = outs.permute(1, 0, 2)  # N*K*C
    # softmax_outs = outs.softmax(dim=-1)  # N*K*C
    num_nodes = softmax_outs.shape[0]
    Pi = (
        mixture_embeds(torch.arange(num_nodes).to(device)).softmax(dim=-1).unsqueeze(-1)
    )  # N*k*1
    # (N*K*1)*(N*K*C).sum(dim=1) => (N*K*C).sum(dim=1) => (N*C)
    softmax_outs = (Pi * softmax_outs).sum(dim=1)
    return softmax_outs


@torch.no_grad()
def test(
    model,
    mixture_embeds,
    probs_all,  # N*K*C
    y,
    train_mask,
    val_mask,
    test_mask,
    device,
):
    model.eval()
    accs = []
    # logits = model.inference(x, edge_index, edge_attr)  # N*K*C
    pred_prob = predict(probs_all, mixture_embeds, device)  # N*C
    torch.set_printoptions(sci_mode=False)
    for mask in [train_mask, val_mask, test_mask]:
        pred = pred_prob[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
def ogb_test(model, mixture_embeds, probs_all, y_true, split_idx, evaluator, device):
    model.eval()
    # logits_all has dimension (N * K * C)
    pred_prob = predict(probs_all, mixture_embeds, device)  # N*C
    y_pred = pred_prob.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    valid_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return [train_acc, valid_acc, test_acc]


def get_dataset(dataset_name):
    if dataset_name.lower() == "wikics":
        path = osp.join(osp.expanduser("~"), "datasets", "WikiCS")
        dataset = WikiCS(path, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "amazon_computers":
        path = osp.join(osp.expanduser("~"), "datasets", "Amazon")
        dataset = Amazon(path, name="Computers", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "amazon_photo":
        path = osp.join(osp.expanduser("~"), "datasets", "Amazon")
        dataset = Amazon(path, name="photo", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "coauthor_cs":
        path = osp.join(osp.expanduser("~"), "datasets", "Coauthor")
        dataset = Coauthor(path, name="cs", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "coauthor_physics":
        path = osp.join(osp.expanduser("~"), "datasets", "Coauthor")
        dataset = Coauthor(path, name="physics", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "cora":
        path = osp.join(osp.expanduser("~"), "datasets", "Cora")
        dataset = Planetoid(path, name="cora", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "citeseer":
        path = osp.join(osp.expanduser("~"), "datasets", "Citeseer")
        dataset = Planetoid(path, name="citeseer", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "pubmed":
        path = osp.join(osp.expanduser("~"), "datasets", "Pubmed")
        dataset = Planetoid(path, name="pubmed", transform=T.NormalizeFeatures())
        data = dataset[0]
    elif dataset_name.lower() == "ogbn-arxiv":
        path = osp.join(osp.expanduser("~"), "datasets")
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv", root=path, transform=T.ToUndirected()
        )
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f"{key}_mask"] = mask
        data["val_mask"] = data["valid_mask"]
        data.y = torch.flatten(data.y)
    elif dataset_name.lower() == "ogbn-products":
        path = osp.join(osp.expanduser("~"), "datasets")
        dataset = PygNodePropPredDataset(name="ogbn-products", root=path)
    elif dataset_name.lower() == "mixed-citation":
        cora_path = osp.join(osp.expanduser("~"), "datasets", "Cora")
        cora_dataset = Planetoid(
            cora_path, name="cora", transform=T.NormalizeFeatures()
        )
        citeseer_path = osp.join(osp.expanduser("~"), "datasets", "Citeseer")
        citeseer_dataset = Planetoid(
            citeseer_path, name="citeseer", transform=T.NormalizeFeatures()
        )
        pubmed_path = osp.join(osp.expanduser("~"), "datasets", "Pubmed")
        pubmed_dataset = Planetoid(
            pubmed_path, name="pubmed", transform=T.NormalizeFeatures()
        )
        print(cora_dataset.__dict__)
        print(citeseer_dataset.__dict__)
        print(pubmed_dataset.__dict__)
        path = osp.join(osp.expanduser("~"), "datasets", "Mixed_Citation")
        dataset = MixedCitation(
            path,
            name="cora_citeseer_pubmed",
            datasets=[cora_dataset, citeseer_dataset, pubmed_dataset],
            transform=T.NormalizeFeatures(),
        )
    else:
        print("Dataset ", dataset_name, "does not exist!")
        import sys

        sys.exit()
    return dataset, data


class MixedCitation(InMemoryDataset):
    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(
        self,
        root: str,
        name: str,
        datasets: list,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name

        super().__init__(root, transform, pre_transform)

        total_num_nodes = sum([dataset.data.x.shape[0] for dataset in datasets])
        total_num_features = sum([dataset.data.x.shape[1] for dataset in datasets])
        total_num_edges = sum(
            [dataset.data.edge_index.shape[1] for dataset in datasets]
        )
        train_mask = torch.concat([dataset.data.train_mask for dataset in datasets], 0)
        val_mask = torch.concat([dataset.data.val_mask for dataset in datasets], 0)
        test_mask = torch.concat([dataset.data.test_mask for dataset in datasets], 0)
        # x = torch.randn(total_num_nodes, total_num_features, dtype=torch.float32)
        x = torch.normal(mean=0, std=0.05, size=(total_num_nodes, total_num_features))
        y = torch.zeros(total_num_nodes, dtype=torch.long)
        edge_index = torch.zeros(2, total_num_edges, dtype=torch.long)
        dataset_ids = torch.zeros(total_num_nodes, dtype=torch.long)

        i = 0  # index for nodes
        j = 0  # index for features
        k = 0  # index for edges
        m = 0  # index for num classes
        for n, dataset in enumerate(datasets):
            num_nodes = dataset.data.x.shape[0]
            num_features = dataset.data.x.shape[1]
            num_edges = dataset.data.edge_index.shape[1]
            num_classes = dataset.num_classes
            x[i : i + num_nodes, j : j + num_features] = dataset.data.x
            y[i : i + num_nodes] = dataset.data.y + m
            edge_index[:, k : k + num_edges] = dataset.data.edge_index + i
            dataset_ids[i : i + num_nodes] = torch.full((num_nodes,), n)
            i += num_nodes
            j += num_features
            k += num_edges
            m += num_classes
        # print(f"{x=}")
        # print(f'{edge_index=}')
        # print(f'{num_nodes=}')
        # print(f'{num_features=}')
        # print(f"{y=}")
        # print(f'{y.shape=}')

        self.data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask
        self.data.dataset_ids = dataset_ids

    def __repr__(self) -> str:
        return f"{self.name}()"


def log_csv(
    unnormalized_mixture_embeds,
    mixed_embeds,
    K_embeds,
    args,
    y,
    train_mask,
    dataset_ids,
):
    args_str = str(args).replace(" ", "")
    today_date = date.today().strftime("%b-%d-%Y")
    exp_path = f"{today_date}/{args_str}"
    os.makedirs(exp_path, exist_ok=True)

    for i, embed in enumerate(K_embeds):
        embeds_df = pd.DataFrame(embed.detach().cpu().numpy())
        embeds_df.to_csv(f"{exp_path}/embeddings_{i}", index=False)

    mixture_embeds_df = pd.DataFrame(
        unnormalized_mixture_embeds.weight.detach().cpu().numpy()
    )
    mixture_embeds_df.to_csv(f"{exp_path}/mixture_embeds", index=False)

    mixture_dist_df = pd.DataFrame(
        unnormalized_mixture_embeds.weight.softmax(dim=-1).detach().cpu().numpy()
    )
    mixture_dist_df.to_csv(f"{exp_path}/mixture_dist", index=False)

    mixed_embeds_df = pd.DataFrame(mixed_embeds.detach().cpu().numpy())
    mixed_embeds_df.to_csv(f"{exp_path}/mixed_embeds", index=False)

    y_df = pd.DataFrame(y.detach().cpu().numpy())
    y_df.to_csv(f"{exp_path}/y", index=False)

    train_mask_df = pd.DataFrame(train_mask.detach().cpu().numpy())
    train_mask_df.to_csv(f"{exp_path}/train_mask", index=False)

    dataset_ids_df = pd.DataFrame(dataset_ids.detach().cpu().numpy())
    dataset_ids_df.to_csv(f"{exp_path}/dataset_ids", index=False)
    return
