import torch
import argparse
from misc.logger import Logger
from misc import gram


def e_step(model, num_nodes, device, num_gcns, x, train_y, mixture_embeds, train_idx):
    print("e_step")

    # Get logits for training nodes
    with torch.no_grad():
        model.eval()
        train_outs = model.get_train_outs(x)

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
    optimizer_pi,
    gamma,
    train_idx,
    edge_index,
    rho,
    num_nodes,
    device,
):
    model.train()
    outs_train = model.get_train_outs(x)  # List of (N_train X C) of size k
    model_loss, pi_loss, jsd_loss = gram.m_step_optimize(
        model,
        mixture_embeds,
        optimizer,
        outs_train,
        y_train,
        train_idx,
        edge_index,
        gamma,
        rho,
        num_nodes,
        device,
    )

    return model_loss, pi_loss, jsd_loss


def get_arg():
    parser = argparse.ArgumentParser(description="XXX")
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
    parser.add_argument("--estep_per_epoch", type=int, default=20)
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_arg()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Get Dataset
    data = None
    x = None
    y_train = None
    train_idx = None
    edge_index = None
    edge_attr = None
    num_nodes = None
    num_gcns = args.num_gcns

    # Define model
    model = None

    logger = Logger(args.runs, args)
    mixture_embeds = gram.MixtureEmbeddings(
        data.num_nodes, args.num_gcns, device
    ).mixture_embeds
    data = data.to(device)

    for run in range(args.runs):
        model.reset_parameters()
        mixture_embeds.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer_pi = torch.optim.Adam(mixture_embeds.parameters(), lr=args.pi_lr)
        for epoch in range(0, args.epochs):
            if epoch % args.estep_per_epoch == 0:
                gamma = e_step(
                    model=model,
                    num_nodes=num_nodes,
                    device=device,
                    num_gcns=num_gcns,
                    x=x,
                    train_y=y_train,
                    mixture_embeds=mixture_embeds,
                    train_id=train_idx,
                )
                Pi = mixture_embeds(torch.arange(num_nodes).to(device)).softmax(dim=-1)
                print(Pi)
            loss, pi_loss, jsd_loss = m_step(
                model=model,
                mixture_embeds=mixture_embeds,
                x=x,
                y_train=y_train,
                optimizer=optimizer,
                optimizer_pi=optimizer_pi,
                gamma=gamma,
                train_idx=data.train_mask,
                edge_index=None,
                rho=args.rho,
                num_nodes=num_nodes,
                device=device,
            )

            result = gram.test(
                model=model,
                mixture_embeds=mixture_embeds,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
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
                    f"Pi Loss: {pi_loss:.4f}, "
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
