import argparse
import os.path as osp

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import SIGN

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from misc.logger import Logger
from misc import gram
from models.sign import MLP


def e_step(
    model, num_nodes, device, num_gcns, sign_xs, train_y, mixture_embeds, train_idx
):
    print("e_step")

    # Get logits for training nodes
    with torch.no_grad():
        model.eval()
        train_outs = model.get_train_outs(sign_xs)  # List of (N_train X C) of size k

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
    xs,
    y_train,
    optimizer,
    gamma,
    train_idx,
    edge_index,
    rho,
    num_nodes,
    device,
):
    model.train()
    outs_train = model.get_train_outs(xs)  # List of (N_train X C) of size k
    model_loss, jsd_loss = gram.m_step_optimize(
        model=model,
        optimizer=optimizer,
        train_outs=outs_train,
        edge_index=edge_index,
        train_mask=train_idx,
        train_y=y_train,
        gamma=gamma,
        rho=rho,
        mixture_embeds=mixture_embeds,
        num_nodes=num_nodes,
        mixture_prop_sample_size=num_nodes * 10,
        device=device,
    )

    return model_loss, jsd_loss


def get_arg():
    parser = argparse.ArgumentParser(description="cora cite seer pubmed SIGN")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--pi_lr",
        type=float,
        default=0.003,
        help="learning rate set for pi optimizer.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num_gcns", type=int, default=1, help="the number of GCNs")
    parser.add_argument("--rho", type=float, default=30.0, help="for jsd_loss")
    parser.add_argument(
        "--dataset", type=str, default="Cora", help="Cora/Citeseer/Pubmed"
    )
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_arg()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", args.dataset)
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = SIGN(args.num_layers)(dataset[0])  # This might take a while.
    num_nodes = data.num_nodes
    num_gcns = args.num_gcns

    xs = [data.x] + [data[f"x{i}"] for i in range(1, args.num_layers + 1)]
    xs = [x.to(device) for x in xs]
    xs_train = [x[data.train_mask] for x in xs]
    train_idx = torch.arange(data.num_nodes)[data.train_mask].to(device)
    # xs_train = [x[split_idx["train"]].to(device) for x in xs]
    # xs_valid = [x[split_idx["valid"]].to(device) for x in xs]
    # xs_test = [x[split_idx["test"]].to(device) for x in xs]

    y_train = data.y[data.train_mask].to(device)
    # y_valid_true = data.y[split_idx["valid"]].to(device)
    # y_test_true = data.y[split_idx["test"]].to(device)
    # train_idx = split_idx["train"].to(device)
    # valid_idx = split_idx["valid"].to(device)
    # test_idx = split_idx["test"].to(device)

    model = MLP(
        in_channels=data.x.size(-1),
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_gcns=args.num_gcns,
        num_nodes=data.num_nodes,
        device=device,
    ).to(device)

    logger = Logger(args.runs, args)
    mixture_embeds = gram.MixtureEmbeddings(num_nodes, num_gcns, device).mixture_embeds
    data = data.to(device)

    for run in range(args.runs):
        model.reset_parameters()
        mixture_embeds.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(0, args.epochs):
            gamma = e_step(
                model=model,
                num_nodes=num_nodes,
                device=device,
                num_gcns=num_gcns,
                sign_xs=xs_train,
                train_y=y_train,
                mixture_embeds=mixture_embeds,
                train_idx=train_idx,
            )
            # Pi = mixture_embeds(torch.arange(num_nodes).to(device)).softmax(dim=-1)
            # print(Pi)
            loss, jsd_loss = m_step(
                model=model,
                mixture_embeds=mixture_embeds,
                xs=xs_train,
                y_train=y_train,
                optimizer=optimizer,
                gamma=gamma,
                train_idx=data.train_mask,
                edge_index=data.edge_index,
                rho=args.rho,
                num_nodes=num_nodes,
                device=device,
            )

            # TEST
            model.eval()
            logits_all = model.inference(xs=xs)  # N*K*C

            result = gram.test(
                model=model,
                mixture_embeds=mixture_embeds,
                logits_all=logits_all,
                y=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                test_mask=data.test_mask,
                device=device,
            )
            logger.add_result(run, result)

            if (epoch % args.log_steps == 0) and (epoch != 0):
                train_acc, valid_acc, test_acc = result
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"JSD Loss: {jsd_loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}%, "
                    f"Test: {100 * test_acc:.2f}%"
                )

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        main()
