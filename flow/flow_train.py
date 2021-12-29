import sys

sys.path.append("../")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle as pickle

from fast_jtnn import *
import rdkit
from tqdm import tqdm
import os

from flow import (
    cnf,
    FlowDataset,
    MaskedAutoregressiveFlow,
    NICE,
    N01_loss,
    naive_loss,
    SimpleRealNVP,
)


def evaluate(test_dataloader, flow, flow_type, conditional, loss_fn, sigma):
    device = next(flow.parameters()).device
    test_metrics, test_steps = {}, 0
    with torch.no_grad():
        for w_tree, w_mol, a in test_dataloader:
            w_tree, w_mol, a = w_tree.to(device), w_mol.to(device), a.to(device)
            w = torch.cat([w_tree, w_mol], dim=1)
            if flow_type == "NICE":
                z, logdet = flow(w)
            elif flow_type == "CNF":
                zero_padding = torch.zeros(1, 1, 1, device=device)
                cond = a.unsqueeze(-1).unsqueeze(-1) if conditional else torch.ones(a.shape[0], 1, 1, 1, device=device)
                z, logdet = flow(w.unsqueeze(1), cond, zero_padding)
                z = z.squeeze(1)
            elif flow_type == "RealNVP" or flow_type == "MAF":
                z, logdet = flow._transform(w, context=a if conditional else None)
            else:
                raise ValueError
            _, metrics = loss_fn(z, logdet) if conditional else loss_fn(z, logdet, a, sigma)
            test_steps += 1
            for k, v in metrics.items():
                test_metrics[k] = (
                    test_metrics[k] + metrics[k].item()
                    if k in test_metrics
                    else metrics[k].item()
                )
    return test_steps, test_metrics


def main_flow_train(
    mol_path,
    property_path,
    vocab,
    save_dir,
    jtvae_path,
    flow_type,
    conditional=False,
    hidden_size=450,
    batch_size=32,
    latent_size=56,
    depthT=20,
    depthG=3,
    flow_load_epoch=0,
    flow_n_layers=4,
    flow_n_blocks=4,
    flow_sigma=1.0,
    flow_sigma_decay=0.98,
    lr=1e-3,
    epochs=50,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    jtvae = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG).to(device)
    loaded = torch.load(jtvae_path, map_location=device)
    jtvae.load_state_dict(loaded)
    jtvae.eval()
    print(f"Loaded pretrained JTNNVAE from {jtvae_path}")

    dataset = FlowDataset(
        mol_path, property_path, vocab, jtvae, save_path=f"{mol_path}/..", load=True
    )
    train_data = Subset(dataset, range(0, 240000))
    test_data = Subset(dataset, range(240000, 246400))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    del jtvae

    if flow_type == "NICE":
        if conditional:
            raise ValueError
        flow = NICE(
            input_dim=latent_size,
            n_layers=flow_n_layers,
            n_couplings=flow_n_blocks,
            hidden_dim=latent_size,
            device=device,
        ).to(device)
    elif flow_type == "CNF":
        flow = cnf(
            latent_size,
            "-".join([str(latent_size)] * (flow_n_layers - 1)),
            1,
            flow_n_blocks,
        ).to(device)

        zero_padding = torch.zeros(1, 1, 1, device=device)
    elif flow_type == "MAF":
        if conditional:
            context_features, embedding_features = 1, None
        else:
            context_features, embedding_features = None, None
        flow = MaskedAutoregressiveFlow(
            latent_size,
            latent_size,
            context_features,
            embedding_features,
            flow_n_layers,
            flow_n_blocks,
        ).to(device)
    elif flow_type == "RealNVP":
        if conditional:
            context_features = 1
        else:
            context_features = None
        flow = SimpleRealNVP(
            latent_size, latent_size, context_features, flow_n_layers, flow_n_blocks
        ).to(device)
    else:
        raise ValueError
    print(flow)

    optimizer = optim.Adam(flow.parameters(), lr=lr)

    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    if flow_load_epoch:
        loaded = torch.load(
            save_dir + "/flow.epoch-" + str(flow_load_epoch), map_location=device
        )
        flow.load_state_dict(loaded["flow"])
        optimizer.load_state_dict(loaded["optimizer"])
        flow.eval()

    loss_fn = N01_loss if conditional else naive_loss

    for epoch in range(flow_load_epoch, epochs):
        flow.train()

        train_steps = 0
        train_metrics = {}
        curr_flow_sigma = flow_sigma * (flow_sigma_decay ** epoch)
        for w_tree, w_mol, a in tqdm(train_dataloader):
            w_tree, w_mol, a = w_tree.to(device), w_mol.to(device), a.to(device)
            w = torch.cat([w_tree, w_mol], dim=1)
            train_steps += 1
            flow.zero_grad()
            if flow_type == "NICE":
                z, logdet = flow(w)
            elif flow_type == "CNF":
                cond = (
                    a.unsqueeze(-1).unsqueeze(-1)
                    if conditional
                    else torch.ones(a.shape[0], 1, 1, 1, device=device)
                )
                z, logdet = flow(w.unsqueeze(1), cond, zero_padding)
                z = z.squeeze(1)
                logdet = -logdet  # it's possible that there should be minus
            elif flow_type == "RealNVP" or flow_type == "MAF":
                z, logdet = flow._transform(w, context=a if conditional else None)
            else:
                raise ValueError
            if conditional:
                loss, metrics = loss_fn(z, logdet)
            else:
                loss, metrics = loss_fn(z, logdet, a, curr_flow_sigma)
            loss.backward()
            optimizer.step()
            for k, v in metrics.items():
                train_metrics[k] = (
                    train_metrics[k] + metrics[k].item()
                    if k in train_metrics
                    else metrics[k].item()
                )
        test_steps, test_metrics = evaluate(
            test_dataloader, flow, flow_type, conditional, loss_fn, curr_flow_sigma
        )
        print(f"Epoch: {epoch}")
        print(
            "Train metrics: ",
            {k: round(v / train_steps, 3) for k, v in train_metrics.items()},
        )
        print(
            "Test metrics: ",
            {k: round(v / test_steps, 3) for k, v in test_metrics.items()},
        )
        sys.stdout.flush()

        torch.save(
            {"flow": flow.state_dict(), "optimizer": optimizer.state_dict()},
            save_dir + "/flow.epoch-" + str(epoch + 1),
        )
    return flow


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument("--mol_path", required=True)
    parser.add_argument("--property_path", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--jtvae_path", type=str, required=True)

    parser.add_argument(
        "--flow_type",
        type=str,
        choices=["CNF", "MAF", "NICE", "RealNVP"],
        required=True,
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)

    parser.add_argument("--flow_load_epoch", type=int, default=0)
    parser.add_argument("--flow_n_layers", type=int, default=4)
    parser.add_argument("--flow_n_blocks", type=int, default=4)
    parser.add_argument("--flow_sigma", type=float, default=1.0)
    parser.add_argument("--flow_sigma_decay", type=float, default=0.98)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    print(args)

    main_flow_train(
        args.mol_path,
        args.property_path,
        args.vocab,
        args.save_dir,
        args.jtvae_path,
        args.flow_type,
        args.conditional,
        args.hidden_size,
        args.batch_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        args.flow_load_epoch,
        args.flow_n_layers,
        args.flow_n_blocks,
        args.flow_sigma,
        args.flow_sigma_decay,
        args.lr,
        args.epochs,
    )
