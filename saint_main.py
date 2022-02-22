import argparse
import os.path as osp

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph
import torch_geometric.transforms as T
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from models import saint

from misc.logger import Logger
from misc import gram
from misc.gram import get_dataset, log_csv

from torch_geometric.datasets import Planetoid


def e_step(
    model, num_nodes, device, num_gcns, loader, train_y, mixture_embeds, train_idx
):
    print("e_step")

    with torch.no_grad():
        model.eval()
        train_outs = model.get_train_outs(
            loader, train_idx
        )  # List of (N_train X C) of size K

    # Get updated Gamma (a N * k tensor), the responsibility
    gamma = gram.e_step_optimize(
        model,
        train_outs,
        train_y,
        mixture_embeds,
        train_idx,
        num_nodes,
        num_gcns,
        device,
    )
    return gamma


def m_step(
    model,
    mixture_embeds,
    loader,
    edge_index,
    optimizer,
    rho,
    gamma,
    num_edge_samples,
    device,
):
    model.train()

    total_loss = 0
    total_jsd_loss = 0
    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        outs = model(batch_data.x, batch_data.edge_index)
        batch_outs = [
            out[batch_data.train_mask] for out in outs
        ]  # List of [N_batch_train * C] of size K
        batch_y = batch_data.y[batch_data.train_mask]
        if len(batch_y) == 0:
            continue
        batch_idx = batch_data.idx[batch_data.train_mask].squeeze(1)
        batch_gamma = gamma[batch_idx]

        batch_loss, batch_jsd_loss = gram.m_step_optimize(
            model=model,
            optimizer=optimizer,
            train_outs=batch_outs,
            edge_index=edge_index,
            train_y=batch_y,
            train_gamma=batch_gamma,
            rho=rho,
            train_mixture_embeds=mixture_embeds(batch_idx),
            mixture_embeds=mixture_embeds,
            mixture_prop_sample_size=num_edge_samples,
            device=device,
        )
        total_loss += batch_loss
        total_jsd_loss += batch_jsd_loss

    return total_loss / len(loader), total_jsd_loss / len(loader)


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(
        mask, data.edge_index, None, relabel_nodes=True, num_nodes=data.num_nodes
    )
    data.num_nodes = mask.sum().item()
    return data


def get_arg():
    parser = argparse.ArgumentParser(description="GraphSAINT")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    # parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num_gcns", type=int, default=1, help="the number of GCNs")
    parser.add_argument(
        "--num_edge_samples",
        type=int,
        default=1000,
        help="the number of sampled edges for the jsd loss",
    )
    parser.add_argument("--rho", type=float, default=30.0, help="rho")
    parser.add_argument("--use_bns", action="store_true")
    parser.add_argument(
        "--dataset", type=str, default="Cora", help="Cora/Citeseer/Pubmed"
    )
    parser.add_argument("--estep_per_epoch", type=int, default=20)
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_arg()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset, data = get_dataset(args.dataset)
    num_nodes = data.num_nodes
    num_gcns = args.num_gcns
    data.idx = torch.arange(num_nodes).unsqueeze(dim=-1).detach()
    train_idx = torch.arange(data.num_nodes)[data.train_mask].to(device)
    y_train = data.y[data.train_mask].to(device)

    # We omit normalization factors here since those are only defined for the
    # inductive learning setup.
    sampler_data = data
    if args.inductive:
        sampler_data = to_inductive(data)

    loader = GraphSAINTRandomWalkSampler(
        sampler_data,
        batch_size=args.batch_size,
        walk_length=args.walk_length,
        num_steps=args.num_steps,
        sample_coverage=0,
        save_dir=dataset.processed_dir,
    )

    model = saint.SAGE(
        in_channels=data.x.size(-1),
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_gcns=num_gcns,
        num_nodes=num_nodes,
        use_bns=args.use_bns,
        device=device,
    ).to(device)

    subgraph_loader = NeighborSampler(
        data.edge_index, sizes=[-1], batch_size=4096, shuffle=False, num_workers=8
    )
    logger = Logger(args.runs, args)
    mixture_embeds = gram.MixtureEmbeddings(num_nodes, num_gcns, device).mixture_embeds
    data = data.to(device)

    for run in range(args.runs):
        model.reset_parameters()
        mixture_embeds.reset_parameters()
        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": mixture_embeds.parameters()}],
            lr=args.lr,
        )
        for epoch in range(0, args.epochs):
            gamma = e_step(
                model=model,
                num_nodes=num_nodes,
                device=device,
                num_gcns=num_gcns,
                loader=loader,
                train_y=y_train,
                mixture_embeds=mixture_embeds,
                train_idx=train_idx,
            )
            Pi = mixture_embeds(torch.arange(num_nodes).to(device)).softmax(dim=-1)
            print(Pi)
            loss, jsd_loss = m_step(
                model=model,
                mixture_embeds=mixture_embeds,
                loader=loader,
                edge_index=data.edge_index,
                optimizer=optimizer,
                rho=args.rho,
                gamma=gamma,
                num_edge_samples=args.num_edge_samples,
                device=device,
            )
            if epoch % args.log_steps == 0:
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"JSD Loss: {jsd_loss:.4f}, "
                )

            # TEST
            if epoch % 10 == 0:
                model.eval()
                probs_all = model.inference(
                    x_all=data.x, subgraph_loader=subgraph_loader, device=device
                )
                if "ogb" in args.dataset.lower():
                    result = gram.ogb_test(
                        model=model,
                        mixture_embeds=mixture_embeds,
                        probs_all=probs_all,
                        y_true=data.y.unsqueeze(-1),
                        split_idx=dataset.get_idx_split(),
                        evaluator=Evaluator(name=args.dataset),
                        device=device,
                    )
                else:
                    result = gram.test(
                        model=model,
                        mixture_embeds=mixture_embeds,
                        probs_all=probs_all,
                        y=data.y,
                        train_mask=data.train_mask,
                        val_mask=data.val_mask,
                        test_mask=data.test_mask,
                        device=device,
                    )
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}% "
                    f"Test: {100 * test_acc:.2f}%"
                )

        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        main()
