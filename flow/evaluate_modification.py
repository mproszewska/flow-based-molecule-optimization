import sys

sys.path.append("../")
import torch
import csv
from torch.utils.data import DataLoader, Subset
import argparse
from typing import List
from fast_jtnn import *
import rdkit
from rdkit.Chem import Crippen

from flow import NICE, FlowDataset
from scores import calculate_logP


def evaluate_flow(flow_path,
                  mol_path,
                  property_path,
                  vocab,
                  jtvae_path,
                  hidden_size=450,
                  batch_size=1,
                  latent_size=56,
                  depthT=20,
                  depthG=3,
                  flow_n_layers=4,
                  flow_n_couplings=4,
                  values=[-2.0, 2.0]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    jtvae = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG).to(device)
    loaded = torch.load(jtvae_path, map_location=device)
    jtvae.load_state_dict(loaded)
    jtvae.eval()
    print(f"Loaded pretrained JTNNVAE from {jtvae_path}")

    dataset = FlowDataset(mol_path, property_path, vocab, jtvae, save_path=f"{mol_path}/..", load=True)
    test_data = Subset(dataset, range(246300, 246400))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    flow = NICE(
        input_dim=latent_size,
        n_layers=flow_n_layers,
        n_couplings=flow_n_couplings,
        hidden_dim=latent_size,
        device=device,
    ).to(device)
    loaded = torch.load(flow_path, map_location=device)
    flow.load_state_dict(loaded["flow"])
    flow.eval()
    print(f"Loaded pretrained NICE from {flow_path}")

    output = dict()
    original_score = []

    get_original_score = True
    for value in values:
        output[str(value)] = []
        with torch.no_grad():
            for w_tree, w_mol, a in test_dataloader:
                w_tree, w_mol, a = w_tree.to(device), w_mol.to(device), a.to(device)

                w = torch.cat([w_tree, w_mol], dim=1)
                z, logdet = flow(w, False)
                z[:, 0] += value
                z_encoded = flow(z, True)

                smiles = jtvae.decode(z_encoded[:, :28], z_encoded[:, 28:], False)
                logP = calculate_logP(smiles)
                output[str(value)].append(logP)

                if get_original_score:
                    original_score.append(a.item())
        get_original_score = False

    output[str(0)] = original_score
    property = property_path.split('/')[-2]
    with open(f"optimization_results/{property}.csv", "w", newline='\n') as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(output.keys())
        writer.writerows(zip(*output.values()))


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument("--flow_path", required=True)
    parser.add_argument("--mol_path", required=True)
    parser.add_argument("--property_path", required=True)
    parser.add_argument("--vocab", required=True)
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
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--print_iter", type=int, default=1000)
    parser.add_argument("--save_iter", type=int, default=5000)

    args = parser.parse_args()
    print(args)

    evaluate_flow(args.flow_path,
                  args.mol_path,
                  args.property_path,
                  args.vocab,
                  args.jtvae_path,
                  )
