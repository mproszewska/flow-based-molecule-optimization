import sys

sys.path.append("../")
import torch
import csv
from torch.utils.data import DataLoader, Subset
import argparse
from fast_jtnn import *
import rdkit

from scores import calculate_logP
from flow import (
    cnf,
    FlowDataset,
    MaskedAutoregressiveFlow,
    NICE,
    SimpleRealNVP,
)


def evaluate_flow(flow_path,
                  mol_path,
                  property_path,
                  vocab,
                  jtvae_path,
                  values,
                  flow_type,
                  conditional=False,
                  update=False,
                  hidden_size=450,
                  batch_size=1,
                  latent_size=56,
                  depthT=20,
                  depthG=3,
                  flow_n_layers=4,
                  flow_n_blocks=4):
    values = [float(value) for value in values]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    jtvae = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG).to(device)
    loaded = torch.load(jtvae_path, map_location=device)
    jtvae.load_state_dict(loaded)
    jtvae.eval()
    print(f"Loaded pretrained JTNNVAE from {jtvae_path}")

    dataset = FlowDataset(mol_path, property_path, vocab, jtvae, save_path=f"{mol_path}/..", load=True)
    test_data = Subset(dataset, range(245400, 246400))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
    
    loaded = torch.load(flow_path, map_location=device)
    flow.load_state_dict(loaded["flow"])
    flow.eval()
    print(f"Loaded pretrained {flow_type} from {flow_path}")

    output = dict()
    original_score = []

    get_original_score = True
    for value in values:
        output[str(value)] = []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                if batch_idx % 10 == 0:
                    print(batch_idx)
                w_tree = data[0]
                w_mol = data[1]
                a = data[2]
                w_tree, w_mol, a = w_tree.to(device), w_mol.to(device), a.to(device)

                w = torch.cat([w_tree, w_mol], dim=1)
                if flow_type == "NICE":
                    z, logdet = flow(w, False)
                    if update:
                        z[:, 0] += value
                    else:
                        z[:,0] = value
                    z_encoded = flow(z, True)
                elif flow_type == "CNF":
                    cond = (
                        a.unsqueeze(-1).unsqueeze(-1)
                        if conditional
                        else torch.ones(a.shape[0], 1, 1, 1, device=device)
                    )
                    z, logdet = flow(w.unsqueeze(1), cond, zero_padding)
                    z = z.squeeze(1)
                    logdet = -logdet  # it's possible that there should be minus
                    if update:
                        z[:, 0] += value
                    else:
                        z[:, 0] = value
                    z_encoded = flow(z, cond, zero_padding, reverse=True)
                elif flow_type == "RealNVP" or flow_type == "MAF":
                    z, logdet = flow._transform(w, context=a if conditional else None)
                    if update:
                        z[:, 0] += value
                    else:
                        z[:, 0] = value
                    z_encoded = flow._transform.inverse(z, context=a if conditional else None)[0]
                else:
                    raise ValueError

                smiles = jtvae.decode(z_encoded[:, :28], z_encoded[:, 28:], False)
                logP = calculate_logP(smiles)
                output[str(value)].append(logP)

                if get_original_score:
                    original_score.append(a.item())
        get_original_score = False

    output[str(0)] = original_score
    property = property_path.split('/')[-2]
    with open(f"optimization_results/{flow_type}{'_cond_' if conditional else '_'}{property}{'_upd' if update else '_set'}.csv", "w",
              newline='\n') as outfile:
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

    parser.add_argument(
        "--flow_type",
        type=str,
        choices=["CNF", "MAF", "NICE", "RealNVP"],
        required=True,
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--update", action="store_true")
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

    parser.add_argument("--values", nargs='+', required=False)

    args = parser.parse_args()
    print(args)

    evaluate_flow(args.flow_path,
                  args.mol_path,
                  args.property_path,
                  args.vocab,
                  args.jtvae_path,
                  args.values,
                  args.flow_type,
                  args.conditional,
                  args.update
                  )
