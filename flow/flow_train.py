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
    Encoder,
    Flow,
    FlowDataset,
    N01_loss,
    naive_loss,
)


def evaluate(test_dataloader, flow, encoder_a, encoder_sc, use_logvar, loss_fn, sigma):
    device = next(flow.parameters()).device
    test_metrics, test_steps = {}, 0
    with torch.no_grad():
        for _, w_tree, w_mol, sc, a in test_dataloader:
            if use_logvar:
                w_tree, _ = w_tree
                w_mol, _ = w_mol
            w_tree, w_mol, sc, a = w_tree.to(device), w_mol.to(device), sc.to(device), a.to(device)
            w = torch.cat([w_tree, w_mol], dim=1)
            encoded_a = encoder_a(a)
            if encoder_sc is not None:
                encoded_sc = encoder_sc(sc)
            else: encoded_sc = None
            z, logdet = flow(w, encoded_a)
            _, metrics = (
                loss_fn(z, logdet, encoded_sc)
                if flow.conditional
                else loss_fn(z, logdet, encoded_a, sigma, encoded_sc)
            )
            test_steps += 1
            for k, v in metrics.items():
                test_metrics[k] = (
                    test_metrics[k] + metrics[k].item()
                    if k in test_metrics
                    else metrics[k].item()
                )
    return test_steps, test_metrics


def sample_w(w_tree, w_tree_logvar, w_mol, w_mol_logvar):
    w_tree = w_tree + torch.randn_like(w_tree_logvar) * torch.exp(0.5 * w_tree_logvar)
    w_mol = w_mol + torch.randn_like(w_mol_logvar) * torch.exp(0.5 * w_mol_logvar)
    return w_tree, w_mol


def main_flow_train(
    jtvae_path,
    smiles_path,
    mol_path,
    scaffold_path,
    attr_path,
    vocab,
    save_dir,
    flow_type,
    conditional=False,
    hidden_size=450,
    batch_size=32,
    latent_size=56,
    depthT=20,
    depthG=3,
    flow_load_epoch=0,
    flow_latent_size=56,
    flow_n_layers=4,
    flow_n_blocks=4,
    flow_sigma=1.0,
    flow_sigma_decay=0.98,
    flow_use_logvar=False,
    encoder_a_identity=False,
    encoder_a_in_features=1,
    encoder_a_out_features=None,
    encoder_a_embedding=False,
    encoder_sc_in_features=1,
    encoder_sc_out_features=None,
    encoder_sc_embedding=False,
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
        smiles_path,
        mol_path,
        scaffold_path,
        attr_path,
        vocab,
        jtvae,
        save_path=f"{mol_path}/..",
        use_logvar=flow_use_logvar,
        load=False,
    )
    print(f"Size of dataset {len(dataset)}")
    test_set_size = 6400 if len(dataset) > 240000 else 1000
    train_data = Subset(dataset, range(0, len(dataset) - test_set_size))
    test_data = Subset(dataset, range(len(dataset) - test_set_size, len(dataset)))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    del jtvae

    flow = Flow(
        flow_type,
        conditional,
        latent_size,
        flow_latent_size,
        flow_n_layers,
        flow_n_blocks,
    ).to(device)
    print(flow)


    encoder_a = Encoder(
        encoder_a_identity, encoder_a_in_features, encoder_a_out_features, encoder_a_embedding
    ).to(device) 
    print(encoder_a)

    if encoder_sc_embedding:
        encoder_sc = Encoder(
            False, encoder_sc_in_features, encoder_sc_out_features, encoder_sc_embedding
        ).to(device)
        print(encoder_sc)
        optimizer = optim.Adam(list(flow.parameters()) + list(encoder_a.parameters()) + list(encoder_sc.parameters()), lr=lr)
    else:
        encoder_sc = None
        optimizer = optim.Adam(list(flow.parameters()) + list(encoder_a.parameters()), lr=lr)



    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    if flow_load_epoch:
        loaded = torch.load(
            save_dir + "/flow.epoch-" + str(flow_load_epoch), map_location=device
        )
        flow.load_state_dict(loaded["flow"])
        if "encoder" in loaded:
            encoder_a.load_state_dict(loaded["encoder"])
            encoder_a.eval()
        if "encoder_a" in loaded:
            encoder_a.load_state_dict(loaded["encoder_a"])
            encoder_a.eval()
        if "encoder_sc" in loaded:
            encoder_sc.load_state_dict(loaded["encoder_sc"])
            encoder_sc.eval()
        optimizer.load_state_dict(loaded["optimizer"])
        flow.eval()

    loss_fn = N01_loss if conditional else naive_loss

    for epoch in range(flow_load_epoch, epochs):
        flow.train()

        train_steps = 0
        train_metrics = {}
        curr_flow_sigma = flow_sigma * (flow_sigma_decay**epoch)
        for _, w_tree, w_mol, sc, a in tqdm(train_dataloader):
            if flow_use_logvar:
                w_tree, w_tree_logvar = w_tree
                w_mol, w_mol_logvar = w_mol

            w_tree, w_mol, sc, a = w_tree.to(device), w_mol.to(device), sc.to(device), a.to(device)
            if flow_use_logvar:
                w_tree, w_mol = sample_w(
                    w_tree, w_tree_logvar.to(device), w_mol, w_mol_logvar.to(device)
                )

            w = torch.cat([w_tree, w_mol], dim=1)
            train_steps += 1
            flow.zero_grad()
            encoded_a = encoder_a(a)
            if encoder_sc is not None:
                encoded_sc = encoder_sc(sc)
            else: encoded_sc = None
            z, logdet = flow(w, encoded_a)

            if flow.conditional:
                loss, metrics = loss_fn(z, logdet, encoded_sc)
            else:
                loss, metrics = loss_fn(z, logdet, encoded_a, curr_flow_sigma, encoded_sc)
            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                train_metrics[k] = (
                    train_metrics[k] + metrics[k].item()
                    if k in train_metrics
                    else metrics[k].item()
                )
        test_steps, test_metrics = evaluate(
            test_dataloader,
            flow,
            encoder_a,
            encoder_sc,
            flow_use_logvar,
            loss_fn,
            curr_flow_sigma,
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
        save_dict = {
                "flow": flow.state_dict(),
                "encoder_a": encoder_a.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        if encoder_sc is not None:
            save_dict["encoder_sc"] = encoder_sc.state_dict()
        if (epoch + 1) % 10 == 0:
            torch.save(
                save_dict,
                save_dir + "/flow.epoch-" + str(epoch + 1),
            )
    return flow


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument("--jtvae_path", type=str, required=True)
    parser.add_argument("--smiles_path", required=True)
    parser.add_argument("--mol_path", required=True)
    parser.add_argument("--scaffold_path", required=True)
    parser.add_argument("--attr_path", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--save_dir", required=True)

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

    parser.add_argument("--flow_latent_size", type=int, default=56)
    parser.add_argument("--flow_n_layers", type=int, default=4)
    parser.add_argument("--flow_n_blocks", type=int, default=4)
    parser.add_argument("--flow_sigma", type=float, default=1.0)
    parser.add_argument("--flow_sigma_decay", type=float, default=0.98)
    parser.add_argument("--flow_use_logvar", action="store_true")

    parser.add_argument("--encoder_a_identity", action="store_true")
    parser.add_argument("--encoder_a_in_features", type=int, default=1)
    parser.add_argument("--encoder_a_out_features", type=int, default=None)
    parser.add_argument("--encoder_a_embedding", action="store_true")
    
    parser.add_argument("--encoder_sc_in_features", type=int, default=1)
    parser.add_argument("--encoder_sc_out_features", type=int, default=None)
    parser.add_argument("--encoder_sc_embedding", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    print(args)

    assert args.flow_use_logvar

    main_flow_train(
        args.jtvae_path,
        args.smiles_path,
        args.mol_path,
        args.scaffold_path,
        args.attr_path,
        args.vocab,
        args.save_dir,
        args.flow_type,
        args.conditional,
        args.hidden_size,
        args.batch_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        args.flow_load_epoch,
        args.flow_latent_size,
        args.flow_n_layers,
        args.flow_n_blocks,
        args.flow_sigma,
        args.flow_sigma_decay,
        args.flow_use_logvar,
        args.encoder_a_identity,
        args.encoder_a_in_features,
        args.encoder_a_out_features,
        args.encoder_a_embedding,
        args.encoder_sc_in_features,
        args.encoder_sc_out_features,
        args.encoder_sc_embedding,
        args.lr,
        args.epochs,
    )
