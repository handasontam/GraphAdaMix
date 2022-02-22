import os.path as osp
import argparse
from tracemalloc import start

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator
from models import graphsage, gcn, gat, mlp

from misc.logger import Logger
from misc import gram
from misc.gram import get_dataset, log_csv
import time


def e_step(
    model,
    num_nodes,
    device,
    num_gcns,
    x,
    edge_index,
    edge_attr,
    train_y,
    mixture_embeds,
    train_idx,
):
    # Get logits for training nodes
    with torch.no_grad():
        model.eval()
        train_outs = model.get_train_outs(
            x, edge_index, edge_attr, train_idx
        )  # List of (N_train X C) of size k

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
    x,
    y_train,
    optimizer,
    gamma,
    train_idx,
    edge_index,
    edge_attr,
    rho,
    num_nodes,
    num_edge_samples,
    device,
):
    model.train()
    outs_train = model.get_train_outs(
        x, edge_index, edge_attr, train_idx
    )  # List of (N_train X C) of size k
    model_loss, jsd_loss = gram.m_step_optimize(
        model=model,
        optimizer=optimizer,
        train_outs=outs_train,
        edge_index=edge_index,
        train_y=y_train,
        train_gamma=gamma[train_idx],
        rho=rho,
        train_mixture_embeds=mixture_embeds(train_idx),
        mixture_embeds=mixture_embeds,
        mixture_prop_sample_size=num_edge_samples,
        device=device,
    )

    return model_loss, jsd_loss


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gdc", action="store_true", help="Use GDC preprocessing.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_gcns", type=int, default=1, help="the number of GCNs")
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--rho", type=float, default=30.0, help="rho")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate set for the model optimizer",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--dataset", type=str, default="Cora", help="Cora/Citeseer/Pubmed"
    )
    parser.add_argument("--num_edge_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--use_bns", action="store_true")
    parser.add_argument("--model", type=str, default="gcn", help="gcn/sage/gat")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_arg()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset, data = get_dataset(args.dataset)
    # data = dataset[0]

    if args.use_gdc:
        gdc = T.GDC(
            self_loop_weight=1,
            normalization_in="sym",
            normalization_out="col",
            diffusion_kwargs=dict(method="ppr", alpha=0.05),
            sparsification_kwargs=dict(method="topk", k=128, dim=0),
            exact=True,
        )
        data = gdc(data)
    data = data.to(device)
    num_nodes = data.num_nodes
    num_gcns = args.num_gcns
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x
    if "ogbn" in args.dataset.lower():
        train_idx = dataset.get_idx_split()["train"].to(device)
        y_train = data.y[train_idx].squeeze().to(device)
        evaluator = Evaluator(name=args.dataset)
    else:
        train_idx = torch.arange(data.num_nodes)[data.train_mask].to(device)
        y_train = data.y[data.train_mask].to(device)

    if args.model.lower() == "sage":
        model = graphsage.GraphSAGE(
            num_gcns,
            num_nodes,
            dataset.num_features,
            dataset.num_classes,
            device,
        ).to(device)
        get_optimizer = graphsage.get_optimizer
    elif args.model.lower() == "gcn":
        model = gcn.PlanetoidGCN(
            num_gcns,
            num_nodes,
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            args.use_gdc,
            device,
        ).to(device)
        get_optimizer = gcn.get_optimizer
    elif args.model.lower() == "gcn_generic":
        model = gcn.GCN(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_gcns=num_gcns,
            num_nodes=num_nodes,
            # use_gdc=args.use_gdc,
            device=device,
        ).to(device)
        get_optimizer = gcn.get_generic_optimizer
    elif args.model.lower() == "gat":
        model = gat.GAT(
            num_gcns,
            num_nodes,
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            device,
        ).to(device)
        get_optimizer = gat.get_optimizer
    elif args.model.lower() == "mlp":
        model = mlp.MLP(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_gcns=num_gcns,
            num_nodes=num_nodes,
            device=device,
        ).to(device)
        get_optimizer = mlp.get_optimizer

    logger = Logger(args.runs, args)

    mixture_embeds = gram.MixtureEmbeddings(num_nodes, num_gcns, device).mixture_embeds

    for run in range(args.runs):
        model.reset_parameters()
        mixture_embeds.reset_parameters()
        best_val_acc = test_acc = 0
        train_acc = val_acc = tmp_test_acc = 0
        optimizer = get_optimizer(args.num_gcns, model, args.lr, mixture_embeds)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_time = time.time()
        for epoch in range(0, args.epochs):
            # E-step: mixture estimation
            gamma = e_step(
                model=model,
                num_nodes=num_nodes,
                device=device,
                num_gcns=num_gcns,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                train_y=y_train,
                mixture_embeds=mixture_embeds,
                train_idx=train_idx,
            )
            # Pi = mixture_embeds(torch.arange(data.num_nodes).to(device)).softmax(dim=-1)
            # print(f"Pi is {Pi}")
            # M-Step: Update model parameters
            model_loss, jsd_loss = m_step(
                model=model,
                mixture_embeds=mixture_embeds,
                x=x,
                y_train=y_train,
                optimizer=optimizer,
                gamma=gamma,
                train_idx=train_idx,
                edge_index=edge_index,
                edge_attr=edge_attr,
                rho=args.rho,
                num_nodes=num_nodes,
                num_edge_samples=args.num_edge_samples,
                device=device,
            )
            # print(model_loss, jsd_loss)
            # TEST
            model.eval()
            outs = model.inference(x, edge_index, edge_attr)
            if "ogbn" in args.dataset.lower():
                train_acc, val_acc, tmp_test_acc = gram.ogb_test(
                    model,
                    mixture_embeds,
                    outs,
                    data.y.to(device),
                    dataset.get_idx_split(),
                    evaluator,
                    device,
                )
            else:
                train_acc, val_acc, tmp_test_acc = gram.test(
                    model=model,
                    mixture_embeds=mixture_embeds,
                    probs_all=outs,  # N*K*C
                    y=data.y,
                    train_mask=data.train_mask,
                    val_mask=data.val_mask,
                    test_mask=data.test_mask,
                    device=device,
                )
            logger.add_result(run, [train_acc, val_acc, tmp_test_acc])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            # log = "Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
            # print(log.format(epoch, train_acc, best_val_acc, test_acc))
        logger.print_statistics(run)
        print("Total time spent: {}".format(time.time() - start_time))
        # softmax_outs =model.inference(x, edge_index, edge_attr)
        # K_embeds = model(x, edge_index, edge_attr)
        # log_csv(
        #     mixture_embeds,
        #     gram.predict(softmax_outs, mixture_embeds, device),
        #     K_embeds,
        #     args,
        #     data.y,
        #     data.train_mask,
        #     data.dataset_ids,
        # )
    logger.print_statistics()


if __name__ == "__main__":
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        main()
