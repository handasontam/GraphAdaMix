import torch
import pandas as pd
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
from torch_geometric.data import dataset
import torch_geometric.transforms as T
import argparse
import math
import numpy as np

from tqdm import tqdm
from torch.optim import AdamW
from GCL.eval import get_split
from misc.bgrl_logistic_regression import LREvaluator
from misc.bgrl_sklearn_logistic_regression import sklearnLREvaluator
from misc.gram import get_dataset
from GCL.models import BootstrapContrast
import torch.optim as optim
from torch_geometric.datasets import WikiCS, Amazon, Coauthor, Planetoid
import misc.scheduler as scheduler
from models.bgrl_gcn import GConv, Encoder
import os

# def e_step(model, x):
#     # Get logits for training nodes
#     with torch.no_grad():
#         model.eval()
#         train_outs = model.get_train_outs(
#             x, edge_index, edge_attr, train_idx
#         )  # List of (N_train X C) of size k


class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum() / b.size(0)
        return b


def train(
    encoder_model, contrast_model, data, tau, optimizer, entropy_criterion, alpha
):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(
        data.x, data.edge_index, data.edge_attr
    )
    loss = contrast_model(
        h1_pred=h1_pred,
        h2_pred=h2_pred,
        h1_target=h1_target.detach(),
        h2_target=h2_target.detach(),
    )
    entropy_loss = entropy_criterion(encoder_model.online_encoder.mixture_embeds.weight)
    (loss + alpha * entropy_loss).backward()
    optimizer.step()
    encoder_model.update_target_encoder(tau)
    return loss.item(), entropy_loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = sklearnLREvaluator()(z, data.y, split)
    return result


def test_citation(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    split = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}
    result = sklearnLREvaluator()(z, data.y, split)
    return result


def get_arg():
    parser = argparse.ArgumentParser(description="BGRL-GRAM")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--pf1", type=float, default=0.2)
    parser.add_argument("--pf2", type=float, default=0.1)
    parser.add_argument("--pe1", type=float, default=0.2)
    parser.add_argument("--pe2", type=float, default=0.3)
    parser.add_argument("--encoder_hidden_dim", type=int, default=512)
    parser.add_argument("--prediction_hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dataset", type=str, default="wikics")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--num_gcns", type=int, default=1, help="the number of GCNs")

    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--rho", type=float, default=30.0, help="for jsd_loss")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="penalty for entropy of the mixture distribution",
    )
    args = parser.parse_args()
    print(args)
    return args


def get_lr_schedule(optimizer, epochs, warmup, last_epoch=-1):
    """adds a lr scheduler to the optimizer.
    :param optimizer: nn.Optimizer
    :returns: scheduler
    :rtype: optim.lr_scheduler
    """
    total_epochs = epochs - warmup
    sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, last_epoch=last_epoch
    )

    # If warmup was requested add it.
    if warmup > 0:
        warmup = scheduler.LinearWarmup(
            optimizer, warmup_steps=warmup, last_epoch=last_epoch
        )
        sched = scheduler.Scheduler(sched, warmup)

    return sched


def get_tau(base_tau, current_epoch, total_epochs):
    return 1 - (
        (1 - base_tau) / 2 * (math.cos(current_epoch * math.pi / total_epochs) + 1)
    )


def log_csv(encoder_model, data, args, y, train_mask):
    args_str = str(args).replace(" ", "")
    exp_path = f"experiments_rebuttal/{args_str}"
    os.makedirs(exp_path, exist_ok=True)

    encoder_model.eval()
    k_embeds_online = encoder_model.online_encoder.forward_k_embed(
        data.x, data.edge_index
    )
    k_embeds_target = encoder_model.target_encoder.forward_k_embed(
        data.x, data.edge_index
    )
    for i, (embed_online, embed_target) in enumerate(
        zip(k_embeds_online, k_embeds_target)
    ):
        z = torch.cat([embed_online, embed_target], dim=1)
        embeds_df = pd.DataFrame(z.detach().cpu().numpy())
        embeds_df.to_csv(f"{exp_path}/embeddings_{i}", index=False)

    mixture_embeds_df = pd.DataFrame(
        encoder_model.online_encoder.mixture_embeds.weight.detach().cpu().numpy()
    )
    mixture_embeds_df.to_csv(f"{exp_path}/mixture_embeds", index=False)

    mixture_dist_df = pd.DataFrame(
        encoder_model.online_encoder.mixture_embeds.weight.softmax(dim=-1)
        .detach()
        .cpu()
        .numpy()
    )
    mixture_dist_df.to_csv(f"{exp_path}/mixture_dist", index=False)

    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    mixed_embeds_df = pd.DataFrame(z.detach().cpu().numpy())
    mixed_embeds_df.to_csv(f"{exp_path}/mixed_embeds", index=False)
    return


def main():
    args = get_arg()
    device = torch.device("cuda:{}".format(args.device))
    dataset, data = get_dataset(args.dataset)
    data = data.to(device, non_blocking=True)
    num_nodes = data.num_nodes
    print(data)

    aug1 = A.Compose([A.EdgeRemoving(pe=args.pe1), A.FeatureMasking(pf=args.pf1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.pe2), A.FeatureMasking(pf=args.pf2)])

    gconv = GConv(
        input_dim=dataset.num_features,
        encoder_hidden_dim=args.encoder_hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=2,
        num_gcns=args.num_gcns,
        num_nodes=num_nodes,
        device=device,
    ).to(device)
    encoder_model = Encoder(
        encoder=gconv,
        augmentor=(aug1, aug2),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.prediction_hidden_dim,
    ).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode="L2L").to(device)
    entropy_criterion = HLoss()

    optimizer = AdamW(encoder_model.parameters(), lr=args.lr, weight_decay=1e-5)
    base_tau = 0.99
    tau = get_tau(base_tau, 0, args.epochs)
    scheduler = get_lr_schedule(optimizer, args.epochs, args.warmup)
    print(encoder_model)

    # encoder_model.update_target_encoder(0)
    # log_csv(encoder_model, data, args)
    # import sys
    # sys.exit()
    with tqdm(total=args.epochs, desc="(T)") as pbar:
        for epoch in range(1, args.epochs + 1):
            loss, entropy_loss = train(
                encoder_model,
                contrast_model,
                data,
                tau,
                optimizer,
                entropy_criterion,
                args.alpha,
            )
            scheduler.step()
            tau = get_tau(base_tau, epoch, args.epochs)
            with torch.no_grad():
                mixture_embeds = encoder_model.online_encoder.mixture_embeds.weight
                entropies = (
                    (
                        -F.softmax(mixture_embeds, dim=1)
                        * F.log_softmax(mixture_embeds, dim=1)
                    )
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                pbar.set_postfix(
                    {
                        "loss": loss,
                        "entropy_loss": entropy_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "tau": tau,
                        "max_entropy": np.max(entropies),
                        "mean_entropy": np.mean(entropies),
                        "min_entropy": np.min(entropies),
                        # "mixture_embeds": gconv.mixture_embeds.weight.softmax(dim=-1),
                    }
                )

            # Log the embeddings and mixture_embeds
            # if epoch % 1000 == 0:
            # log_csv(encoder_model, data, args)
            pbar.update()

    if args.dataset.lower() in ["cora", "citeseer", "pubmed"]:
        test_result = test_citation(encoder_model, data)
    else:
        test_result = test(encoder_model, data)
    print(
        f'(E): Best test Acc={test_result["acc"]:.4f}, F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}'
    )


if __name__ == "__main__":
    main()
