import sys

sys.path.append("../")
import torch
import csv
from torch.utils.data import DataLoader, Subset
import argparse
from fast_jtnn import *
import rdkit
from tqdm import tqdm

from scores import *
from flow import (
    FlowDataset,
    Flow,
)


def evaluate_flow(
    flow_path,
    smiles_path,
    mol_path,
    property_path,
    vocab,
    jtvae_path,
    values,
    flow_type,
    conditional=False,
    use_logvar=False,
    generate=False,
    generate_sigma=1.0,
    hidden_size=450,
    batch_size=1,
    latent_size=56,
    depthT=20,
    depthG=3,
    flow_latent_size=56,
    flow_n_layers=4,
    flow_n_blocks=4,
    prefix="",
):
    values = [float(value) for value in values]
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
        property_path,
        vocab,
        jtvae,
        save_path=f"{mol_path}/..",
        load=True,
    )
    test_data = Subset(dataset, range(240000, 246400))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    property = property_path.split("/")[-2 if property_path.endswith("/") else -1]
    calculate_stat = {
        "logP": calculate_logP,
        "SAS": calculate_sas,
        "qed": calculate_qed,
    }[property]

    flow = Flow(
        flow_type,
        conditional,
        latent_size,
        flow_latent_size,
        flow_n_layers,
        flow_n_blocks,
    ).to(device)
    epoch = flow_path.split("-")[-1]
    loaded = torch.load(flow_path, map_location=device)
    flow.load_state_dict(loaded["flow"])
    flow.eval()
    print(f"Loaded pretrained {flow_type} from {flow_path}")

    output = dict()
    original_score = []

    get_original_score = not generate
    for value in values:
        output[str(value)] = []
        output[f"smiles_{value}"] = []
        if not generate:
            output[f"similarity_{value}"] = []

    with torch.no_grad():
        for batch_idx, data in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):
            s, w_tree, w_mol, a = data
            w_tree, w_mol, a = w_tree.to(device), w_mol.to(device), a.to(device)
            if generate:
                w_tree, w_mol = (
                    generate_sigma * torch.randn(w_tree.shape).to(device),
                    generate_sigma * torch.randn(w_mol.shape).to(device),
                )
            w = torch.cat([w_tree, w_mol], dim=1)
            for value in values:
                z, logdet = flow(w, a, reverse=False)

                if flow.conditional:
                    a_new[:] = value * torch.ones_like(a).to(device)
                    w_encoded, _ = flow(z, new_a, reverse=True)
                else:
                    z[:, 0] = value
                    w_encoded, _ = flow(z, None, reverse=True)

                smiles = jtvae.decode(w_encoded[:, :28], w_encoded[:, 28:], False)
                logP = calculate_stat(smiles)
                output[str(value)].append(logP)
                output[f"smiles_{value}"].append(smiles)
                if not generate:
                    similarity = rdkit.DataStructs.FingerprintSimilarity(
                        rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(smiles)),
                        rdkit.Chem.RDKFingerprint(rdkit.Chem.MolFromSmiles(s[0])),
                    )
                    output[f"similarity_{value}"].append(similarity)
                if get_original_score:
                    original_score.append(a.item())
        get_original_score = False

    if get_original_score:
        output["original"] = original_score

    property = property_path.split("/")[-2]
    with open(
        f"outputs/{prefix}{flow_type}{'_cond_' if conditional else '_'}{'logvar_' if use_logvar else ''}{property}{'_gen_' if generate else '_mod_'}e{epoch}.csv",
        "w",
        newline="\n",
    ) as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(output.keys())
        writer.writerows(zip(*output.values()))


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument("--flow_path", required=True)
    parser.add_argument("--smiles_path", required=True)
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
    parser.add_argument("--flow_use_logvar", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generate_sigma", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)
    parser.add_argument("--flow_latent_size", type=int, default=56)
    parser.add_argument("--flow_n_layers", type=int, default=4)
    parser.add_argument("--flow_n_blocks", type=int, default=4)
    parser.add_argument("--values", nargs="+", required=False)
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()
    print(args)

    evaluate_flow(
        args.flow_path,
        args.smiles_path,
        args.mol_path,
        args.property_path,
        args.vocab,
        args.jtvae_path,
        args.values,
        args.flow_type,
        args.conditional,
        args.flow_use_logvar,
        args.generate,
        args.generate_sigma,
        args.hidden_size,
        args.batch_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        args.flow_latent_size,
        args.flow_n_layers,
        args.flow_n_blocks,
        args.prefix,
    )