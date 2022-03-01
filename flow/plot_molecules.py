import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import argparse


def plot_molecules(results_path, value, n=18):
    property = results_path.split("_")[-4]
    flow_type = results_path.split("/")[1].split("_")[0]
    results = pd.read_csv(results_path)

    idx_sorted = (results[str(value)] - value).abs().argsort()[:n]
    values = results[str(value)].iloc[idx_sorted]
    molecules = results[f"smiles_{value}"].iloc[idx_sorted]
    values = list(np.around(np.array(values), 4))

    im = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(smi) for smi in molecules],
        molsPerRow=6,
        subImgSize=(400, 400),
        legends=[str(v) for v in values],
    )

    output_path = f"{results_path[:-4]}_mol.png"
    im.save(output_path)


parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, required=True)
parser.add_argument("--value", type=float, required=True)
parser.add_argument("--n", type=int, default=18)

args = parser.parse_args()

plot_molecules(args.results_path, args.value, args.n)
