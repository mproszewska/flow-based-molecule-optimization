import sys

sys.path.append("../")
import os
import json
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader, Subset
import argparse
from fast_jtnn import *
import rdkit
from tqdm import tqdm
import pandas as pd

from scores import *
from flow import (
    Encoder,
    FlowDataset,
    Flow,
)

def tensorize(smiles, assm=True):
    print(smiles, smiles.__class__)
    if isinstance(smiles, float) and np.isnan(smiles):
        smiles = ""
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

def evaluate_flow(
    flow_path,
    smiles_path,
    mol_path,
    scaffold_path,
    attr_path,
    vocab,
    jtvae_path,
    values,
    flow_type,
    conditional=False,
    use_logvar=False,
    shift=False,
    sample=False,
    sample_sigma=1.0,
    generate="",
    generate_sigma=1.0,
    hidden_size=450,
    batch_size=1,
    latent_size=56,
    depthT=20,
    depthG=3,
    flow_latent_size=56,
    flow_n_layers=4,
    flow_n_blocks=4,
    encoder_a_identity=False,
    encoder_a_in_features=1,
    encoder_a_out_features=None,
    encoder_a_embedding=False,
    encoder_sc_in_features=1,
    encoder_sc_out_features=None,
    encoder_sc_embedding=False,
    output_dir="outputs",
):
    torch.manual_seed(123)
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
        scaffold_path,
        attr_path,
        vocab,
        jtvae,
        save_path=f"{mol_path}/..",
        load=True,
    )
    test_set_size = 5000 if len(dataset) > 240000 else 500
    test_data = Subset(dataset, range(len(dataset) - test_set_size, len(dataset)))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    attr = attr_path.split("/")[-2 if attr_path.endswith("/") else -1]
    with open(f"{mol_path}/../scaffold_dict.json", "r") as f:
        scaffold_dict = json.load(f)
    calculate_stat = {
        "logP": calculate_logP,
        "SAS": calculate_sas,
        "qed": calculate_qed,
        "aromatic_rings": calculate_aromatic_rings,
        "contains_cf3": None,
        "contains_cl": None,
        "contains_cn": None,
        "contains_i": None,
        "contains_f": None,
        "contains_cn_or_cf3": None,
        "contains_any_cl_f_i_cn": None,
        "scaffold_one_hot": lambda x: calculate_scaffold_code(x, scaffold_dict),
        "fingerprint": calculate_fingerprint,
        "fingerprint_1024": calculate_fingerprint,
        "fingerprint_167": calculate_fingerprint,
    }[attr]

    flow = Flow(
        flow_type,
        conditional,
        latent_size,
        flow_latent_size,
        flow_n_layers,
        flow_n_blocks,
    ).to(device)

    encoder_a = Encoder(
        encoder_a_identity,
        encoder_a_in_features,
        encoder_a_out_features,
        encoder_a_embedding,
    ).to(device)
    print(encoder_a)

    if encoder_sc_embedding:
        encoder_sc = Encoder(
            False, encoder_sc_in_features, encoder_sc_out_features, encoder_sc_embedding
        ).to(device)
        print(encoder_sc)
    else:
        encoder_sc = None

    intermediate_steps = False
    epoch = flow_path.split("-")[-1]
    loaded = torch.load(flow_path, map_location=device)
    flow.load_state_dict(loaded["flow"])
    flow.eval()
    if "encoder" in loaded:
        encoder_a.load_state_dict(loaded["encoder"])
        encoder_a.eval()
    else:
        encoder_a.load_state_dict(loaded["encoder_a"])
        if "encoder_sc" in loaded:
            encoder_sc.load_state_dict(loaded["encoder_sc"])

    print([param.data for param in encoder_a.parameters()])
    print(f"Loaded pretrained {flow_type} from {flow_path}")

    prefix = flow_path.split("/")[0]
    
    output = dict()
    original_score = []

    if not generate and (not attr.startswith("fingerprint")):
        output["original"] = []
        output["smiles_original"] = []
    for value in values:
        output[str(value)] = []
        output[f"smiles_{value}"] = []
        if not generate:
            output[f"similarity_{value}"] = []

    with torch.no_grad():
        if generate == "jtvae"  and shift and calculate_stat is None:
            smiles_list, w_trees, w_mols, ass = list(), list(), list(), list()
            for batch_idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                w_tree, w_mol = (
                        generate_sigma * torch.randn([1, 28]).to(device),
                        generate_sigma * torch.randn([1, 28]).to(device),
                        )
                w_trees += [w_tree]
                w_mols += [w_mol]
                smiles_list += [jtvae.decode(w_tree, w_mol, False)]
            print(smiles_list)
            smiles = pd.DataFrame(smiles_list)
            tmp_path = f"tmp_file.csv"
            smiles.to_csv(tmp_path, header=False, index=False)
            group = {
                "contains_cf3": "C(F)(F)F",
                "contains_cl": "[!#1]Cl",
                "contains_f": "[!#1]F",
                "contains_i": "[!#1]I",
                "contains_cn": "C#N",
                "contains_cn_or_cf3": "C(F)(F)F",
                "contains_any_cl_f_i_cn": "[$([!#1]Cl),$([!#1]F),$([!#1]I),$(C#N)]",
            }[attr]
            os.system(
                f"obabel -ismi {tmp_path} -s'{group}' -osmi -O {prefix}_tmp_smiles.smi"
            )

            try:
                positive_smiles = (
                    pd.read_csv(f"{prefix}_tmp_smiles.smi", header=None)[0]
                    .str.strip()
                    .tolist()
                )
                positive_smiles = set(
                    [rdkit.Chem.CanonSmiles(s) for s in positive_smiles]
                )
            except pd.errors.EmptyDataError:
                positive_smiles = set()
            os.remove(f"{prefix}_tmp_smiles.smi")
            if attr == "contains_cn_or_cf3":
                os.system(
                    f"obabel -ismi {tmp_path} -s'C#N' -osmi -O {prefix}_tmp_smiles_neg.smi"
                )
                try:
                    negative_smiles = (
                        pd.read_csv(f"{prefix}_tmp_smiles_neg.smi", header=None)[0]
                        .str.strip()
                        .tolist()
                    )
                    negative_smiles = set(
                        [rdkit.Chem.CanonSmiles(s) for s in negative_smiles]
                    )

                except pd.errors.EmptyDataError:
                    negative_smiles = set()
                os.remove(f"{prefix}_tmp_smiles_neg.smi")
                for s in smiles_list:
                    s = rdkit.Chem.CanonSmiles(s)
                    if (s in positive_smiles) and (s not in negative_smiles):
                        ass += [1]
                    elif (s not in positive_smiles) and (s in negative_smiles):
                        ass += [0]
                    else:
                        ass += [-1]
            else:
                for s in smiles_list:
                    print(s)
                    ass += [ int(rdkit.Chem.CanonSmiles(s) in positive_smiles)]
            ass = torch.tensor(ass).unsqueeze(-1)
        for batch_idx, data in tqdm(
            enumerate(test_dataloader), total=len(test_dataloader)
        ):

            s, w_tree, w_mol, sc, a = data
            w_tree, w_mol, sc, a = (
                w_tree.to(device),
                w_mol.to(device),
                sc.to(device),
                a.to(device),
            )
            if generate == "jtvae" and shift:
                if calculate_stat is None:
                    w_tree, w_mol = w_trees[batch_idx].to(device), w_mols[batch_idx].to(device)
                    a = ass[batch_idx:batch_idx+1].to(device)
                    if a[0, 0] < 0 : continue
                else: 
                    w_tree, w_mol = (
                        generate_sigma * torch.randn(w_tree.shape).to(device),
                        generate_sigma * torch.randn(w_mol.shape).to(device),
                    )
                    smiles = jtvae.decode(w_tree, w_mol, False)
                    a = torch.tensor([calculate_stat(smiles)]).unsqueeze(0).to(device)
            w = torch.cat([w_tree, w_mol], dim=1)
            uq_values = list()
            for i, value in enumerate(values):
                if attr.startswith("fingerprint"):
                    value = dataset.A[int(value)]
                    uq_values += [value]

                encoded_a = encoder_a(a)
                if generate != "flow":
                    z, logdet = flow(w, encoded_a, reverse=False)
                else:
                    z = generate_sigma * torch.randn(w.shape).to(device)
                    if encoder_sc is not None:
                        encoded_sc = encoder_sc(sc)
                        noise = generate_sigma * torch.randn(encoded_sc.shape).to(
                                device
                            )
                        z[:, encoded_a.shape[1] :] = encoded_sc + noise

                encoded_value = encoder_a(value * torch.ones_like(a, device=device))
                if flow.conditional:
                    a_new[:] = value * torch.ones_like(a).to(device)
                    w_encoded, _ = flow(z, new_a, reverse=True)
                else:
                    if len(encoded_value.shape) == 3:
                        encoded_value = encoded_value[:, 0]
                    if shift:
                        shift_vec = encoded_value - encoded_a
                        z[:, 0 : encoded_value.shape[1]] = (
                                z[:, 0 : encoded_value.shape[1]] + shift_vec
                            )
                    elif sample:
                        z[
                                :, 0 : encoded_value.shape[1]
                            ] = encoded_value + sample_sigma * torch.randn(
                                encoded_value.shape
                            ).to(
                                device
                            )
                    else:
                        z[:, 0 : encoded_value.shape[1]] = encoded_value
                    w_encoded, _ = flow(z, None, reverse=True)
                smiles = jtvae.decode(w_encoded[:, :28], w_encoded[:, 28:], False)
                output[f"smiles_{values[i]}"] += [smiles]
                if calculate_stat is not None:
                    output[str(values[i])] += [calculate_stat(smiles)]
                if not generate:
                    output[f"similarity_{values[i]}"] += [
                        calculate_similarity(smiles, s[0])
                    ]

                if not generate and i == 0 and (not attr.startswith("fingerprint")):
                    output["original"] += [a.float().item()]
                    output["smiles_original"] += [s[0]]

    if calculate_stat is None:
        for value in values:
            smiles = pd.DataFrame(output[f"smiles_{value}"])
            tmp_path = f"tmp_file_{value}.csv"
            smiles.to_csv(tmp_path, header=False, index=False)
            group = {
                "contains_cf3": "C(F)(F)F",
                "contains_cl": "[!#1]Cl",
                "contains_f": "[!#1]F",
                "contains_i": "[!#1]I",
                "contains_cn": "C#N",
                "contains_cn_or_cf3": "C(F)(F)F",
                "contains_any_cl_f_i_cn": "[$([!#1]Cl),$([!#1]F),$([!#1]I),$(C#N)]",
            }[attr]
            os.system(
                f"obabel -ismi {tmp_path} -s'{group}' -osmi -O {prefix}_tmp_smiles.smi"
            )

            try:
                positive_smiles = (
                    pd.read_csv(f"{prefix}_tmp_smiles.smi", header=None)[0]
                    .str.strip()
                    .tolist()
                )
                positive_smiles = set(
                    [rdkit.Chem.CanonSmiles(s) for s in positive_smiles]
                )
            except pd.errors.EmptyDataError:
                positive_smiles = set()
            os.remove(f"{prefix}_tmp_smiles.smi")
            if attr == "contains_cn_or_cf3":
                os.system(
                    f"obabel -ismi {tmp_path} -s'C#N' -osmi -O {prefix}_tmp_smiles_neg.smi"
                )
                try:
                    negative_smiles = (
                        pd.read_csv(f"{prefix}_tmp_smiles_neg.smi", header=None)[0]
                        .str.strip()
                        .tolist()
                    )
                    negative_smiles = set(
                        [rdkit.Chem.CanonSmiles(s) for s in negative_smiles]
                    )

                except pd.errors.EmptyDataError:
                    negative_smiles = set()
                os.remove(f"{prefix}_tmp_smiles_neg.smi")
                for s in output[f"smiles_{value}"]:
                    s = rdkit.Chem.CanonSmiles(s)
                    if (s in positive_smiles) and (s not in negative_smiles):
                        output[str(value)] += [1]
                    elif (s not in positive_smiles) and (s in negative_smiles):
                        output[str(value)] += [0]
                    else:
                        output[str(value)] += [None]
            else:

                for s in output[f"smiles_{value}"]:
                    output[str(value)] += [
                        int(rdkit.Chem.CanonSmiles(s) in positive_smiles)
                    ]
    output = pd.DataFrame(output) 
    prefix = flow_path.split("/")[1]
    filename = f"{output_dir}/{prefix}_molecules{f'_gen_{generate}_' if generate is not None else '_mod_'}{'shift_' if shift else ''}{'isteps' if intermediate_steps else ''}e{epoch}.csv"
    output.to_csv(filename, index=False)
    metrics = {
        "value": values,
        "validity": [],
        "mse": [],
        "diversity": [],
    }
    if attr.startswith("fingerprint"):
        for i, v1 in enumerate(uq_values):
            for v2 in uq_values[i + 1 :]:
                print("v1-v2", torch.abs(v1 - v2).sum(), v1.shape, v2.shape)
    if not generate:
        metrics["similarity"] = []
        metrics["non-identity"] = []
    if attr.startswith("contains") or attr in [
        "aromatic_rings",
        "scaffold_one_hot",
        "fingerprint",
        "fingerprint_1024",
        "fingerprint_167",
    ]:
        metrics["accuracy"] = []
    for i, value in enumerate(values):
        if generate or attr.startswith("fingerprint"):
            idx = np.ones((len(output),), dtype=bool)
        else:
            idx = (1 - np.isclose(output["original"], value, atol=0.2)).astype(bool)
        if attr == "contains_cn_or_cf3":
            metrics["validity"] += [(output[str(value)][idx].notna()).mean()]
            idx = idx & output[str(value)].notna()
        else:
            metrics["validity"] += [1.0]
        if not generate:
            metrics["similarity"] += [np.mean(output[f"similarity_{value}"][idx])]
            metrics["non-identity"] += [
                np.mean(output[f"similarity_{value}"][idx] != 1.0)
            ]
        if attr.startswith("fingerprint"):
            metrics["mse"] += [
                np.mean(
                    np.abs(
                        np.stack(output[str(value)][idx]) - uq_values[i].cpu().numpy()
                    )
                    ** 2
                )
            ]
        else:
            metrics["mse"] += [
                np.mean(np.abs(np.stack(output[str(value)][idx]) - value) ** 2)
            ]

        metrics["diversity"] += [calculate_diversity(output[f"smiles_{value}"][idx])]
        if attr.startswith("contains") or attr in [
            "aromatic_rings",
            "scaffold_one_hot",
            "fingerprint",
            "fingerprint_1024",
            "fingerprint_167",
        ]:
            if attr.startswith("fingerprint"):
                metrics["accuracy"] += [
                    np.mean(
                        np.stack(output[str(value)][idx]) == uq_values[i].cpu().numpy()
                    )
                ]
            else:
                metrics["accuracy"] += [
                    np.mean(np.array(output[str(value)][idx]) == value)
                ]

        line = f"value {value} | validity {metrics['validity'][-1]:.4f} | MSE {metrics['mse'][-1]:.4f} | diversity {metrics['diversity'][-1]:.4f}"
        if attr.startswith("contains") or attr in [
            "aromatic_rings",
            "scaffold_one_hot",
            "fingerprint",
            "fingerprint_1024",
            "fingerprint_167",
        ]:
            line += f" | accuracy {metrics['accuracy'][-1]:.4f}"
        if not generate:
            line += f" | similarity {metrics['similarity'][-1]:.4f} | non-identity {metrics['non-identity'][-1]:.4f}"
        print(line)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(
        f"{output_dir}/{prefix}_metrics{f'_gen_{generate}_' if generate is not None else '_mod_'}{'shift_' if shift else ''}{'isteps_' if intermediate_steps else ''}e{epoch}.csv",
        index=False,
    )


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument("--flow_path", required=True)
    parser.add_argument("--smiles_path", required=True)
    parser.add_argument("--mol_path", required=True)
    parser.add_argument("--scaffold_path", required=True)
    parser.add_argument("--attr_path", required=True)
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
    parser.add_argument("--shift", action="store_true")

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample_sigma", type=float, default=0.5)
    parser.add_argument("--generate", type=str, choices=["jtvae", "flow"])
    parser.add_argument("--generate_sigma", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)

    parser.add_argument("--flow_latent_size", type=int, default=56)
    parser.add_argument("--flow_n_layers", type=int, default=4)
    parser.add_argument("--flow_n_blocks", type=int, default=4)

    parser.add_argument("--encoder_a_identity", action="store_true")
    parser.add_argument("--encoder_a_in_features", type=int, default=1)
    parser.add_argument("--encoder_a_out_features", type=int, default=None)
    parser.add_argument("--encoder_a_embedding", action="store_true")

    parser.add_argument("--encoder_sc_in_features", type=int, default=1)
    parser.add_argument("--encoder_sc_out_features", type=int, default=None)
    parser.add_argument("--encoder_sc_embedding", action="store_true")

    parser.add_argument("--values", nargs="+", required=False)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()
    print(args)

    evaluate_flow(
        args.flow_path,
        args.smiles_path,
        args.mol_path,
        args.scaffold_path,
        args.attr_path,
        args.vocab,
        args.jtvae_path,
        args.values,
        args.flow_type,
        args.conditional,
        args.flow_use_logvar,
        args.shift,
        args.sample,
        args.sample_sigma,
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
        args.encoder_a_identity,
        args.encoder_a_in_features,
        args.encoder_a_out_features,
        args.encoder_a_embedding,
        args.encoder_sc_in_features,
        args.encoder_sc_out_features,
        args.encoder_sc_embedding,
        args.output_dir,
    )
