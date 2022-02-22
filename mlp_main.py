import argparse

import torch
import torch.nn.functional as F

from torch.nn import Embedding
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from misc.logger import Logger
from models import mlp
from misc import gram
from misc.gram import get_dataset


def e_step(model, num_nodes, device, x, train_idx, train_y, mixture_embeds, num_gcns):
    # list of nll for each model
    print("e_step")

    # Get logits for training nodes
    with torch.no_grad():
        model.eval()
        train_outs = model(x[train_idx])  # list of [N_train * C] of size K

    # Get updated Gamma (a N * K tensor), the responsibility
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
    x,
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
    outs_train = model(x[train_idx])  # List of (N_train X C) of size k
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
    parser = argparse.ArgumentParser(description="OGBN-Arxiv (MLP)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--use_node_embedding", action="store_true")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num_gcns", type=int, default=1, help="the number of GCNs")
    parser.add_argument("--rho", type=float, default=30.0, help="for jsd_loss")
    parser.add_argument(
        "--pi_lr",
        type=float,
        default=0.03,
        help="learning rate set for pi optimizer.",
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

    x = data.x
    if args.use_node_embedding:
        embedding = torch.load("embedding.pt", map_location="cpu")
        x = torch.cat([x, embedding], dim=-1)
    x = x.to(device)

    y = data.y.to(device)

    if "ogbn" in args.dataset.lower():
        train_idx = dataset.get_idx_split()["train"].to(device)
        y_train = data.y[train_idx].squeeze().to(device)
        evaluator = Evaluator(name=args.dataset)
    else:
        train_idx = torch.arange(data.num_nodes)[data.train_mask].to(device)
        y_train = data.y[data.train_mask].to(device)
    # train_idx = split_idx["train"].to(device)
    # y_train = data.y[data.train_mask].to(device)

    edge_index = data.edge_index.to(device)

    model = mlp.MLP(
        in_channels=x.size(-1),
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_gcns=num_gcns,
        num_nodes=num_nodes,
        device=device,
    ).to(device)

    evaluator = Evaluator(name="ogbn-arxiv")
    logger = Logger(args.runs, args)
    mixture_embeds = gram.MixtureEmbeddings(num_nodes, num_gcns, device).mixture_embeds

    for run in range(args.runs):
        model.reset_parameters()
        mixture_embeds.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(0, args.epochs):
            gamma = e_step(
                model=model,
                num_nodes=num_nodes,
                device=device,
                x=x,
                train_idx=train_idx,
                train_y=y_train,
                mixture_embeds=mixture_embeds,
                num_gcns=num_gcns,
            )
            Pi = mixture_embeds(torch.arange(num_nodes).to(device)).softmax(dim=-1)
            print(Pi)

            loss, jsd_loss = m_step(
                model=model,
                mixture_embeds=mixture_embeds,
                x=x,
                y_train=y_train,
                optimizer=optimizer,
                gamma=gamma,
                train_idx=train_idx,
                edge_index=edge_index,
                rho=args.rho,
                num_nodes=num_nodes,
                device=device,
            )

            # TEST
            model.eval()
            outs = model.inference(x)  # N*K*C
            # gram.ogb_test(model, mixture_embeds, outs, y, split_idx, evaluator, device)
            result = gram.ogb_test(
                model, mixture_embeds, outs, y, split_idx, evaluator, device
            )
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
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
