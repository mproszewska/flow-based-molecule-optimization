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

from flow import NICE, naive_loss, FlowDataset


def main_flow_train(
    mol_path,
    property_path,
    vocab,
    save_dir,
    jtvae_path,
    hidden_size=450,
    batch_size=32,
    latent_size=56,
    depthT=20,
    depthG=3,
    flow_load_epoch=0,
    flow_n_layers=4,
    flow_n_couplings=4,
    flow_sigma=1.0,
    flow_sigma_decay=0.98,
    lr=1e-4,
    epoch=50,
    print_iter=1000,
    save_iter=5000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    jtvae = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG).to(device)
    loaded = torch.load(jtvae_path, map_location=device)
    jtvae.load_state_dict(loaded)
    jtvae.eval()
    print(f"Loaded pretrained JTNNVAE from {jtvae_path}")

    dataset = FlowDataset(mol_path, property_path, vocab, jtvae, save_path=f"{mol_path}/..", load=True)
    train_data = Subset(dataset, range(0, 240000))
    test_data = Subset(dataset, range(240000, 246400))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    del jtvae

    flow = NICE(
        input_dim=latent_size,
        n_layers=flow_n_layers,
        n_couplings=flow_n_couplings,
        hidden_dim=latent_size,
        device=device,
    ).to(device)
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

    total_step = flow_load_epoch
    total_loss = 0.0
    total_metrics = {}
    loss_fn = naive_loss

    for epoch in tqdm(range(epoch)):
        flow.train()
        curr_flow_sigma = flow_sigma * (flow_sigma_decay ** epoch)
        for w_tree, w_mol, a in train_dataloader:
            w_tree, w_mol, a = w_tree.to(device), w_mol.to(device), a.to(device)
            w = torch.cat([w_tree, w_mol], dim=1)
            total_step += 1
            flow.zero_grad()
            z, logdet = flow(w)
            loss, metrics = loss_fn(z, logdet, a, curr_flow_sigma)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = (
                    total_metrics[k] + metrics[k].item()
                    if k in total_metrics
                    else metrics[k].item()
                )

            if total_step % print_iter == 0:
                print(
                    "Epoch %d | Total step %d | Loss: %.3f | Sigma: %.3f"
                    % (epoch, total_step, total_loss / total_step, curr_flow_sigma)
                )
                print({k: round(v / total_step, 3) for k, v in total_metrics.items()})
                sys.stdout.flush()

            if total_step % save_iter == 0:
                torch.save(
                    {"flow": flow.state_dict(), "optimizer": optimizer.state_dict()},
                    save_dir + "/flow.iter-" + str(total_step),
                )
        torch.save(
            {"flow": flow.state_dict(), "optimizer": optimizer.state_dict()},
            save_dir + "/flow.epoch-" + str(epoch+1),
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

    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)

    parser.add_argument("--flow_load_epoch", type=int, default=0)
    parser.add_argument("--flow_n_layers", type=int, default=4)
    parser.add_argument("--flow_n_couplings", type=int, default=4)
    parser.add_argument("--flow_sigma", type=float, default=1.0)
    parser.add_argument("--flow_sigma_decay", type=float, default=0.98)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--print_iter", type=int, default=1000)
    parser.add_argument("--save_iter", type=int, default=5000)

    args = parser.parse_args()
    print(args)

    main_flow_train(
        args.mol_path,
        args.property_path,
        args.vocab,
        args.save_dir,
        args.jtvae_path,
        args.hidden_size,
        args.batch_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        args.flow_load_epoch,
        args.flow_n_layers,
        args.flow_n_couplings,
        args.flow_sigma,
        args.flow_sigma_decay,
        args.lr,
        args.epoch,
        args.print_iter,
        args.save_iter,
    )
