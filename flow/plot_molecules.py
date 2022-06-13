import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
import rdkit
import argparse

def plot_molecules(results_path, value, n=4):
    property = results_path.split("_")[-4]
    flow_type = results_path.split("/")[1].split("_")[0]
    results = pd.read_csv(results_path)

    idx_sorted = (results[f"similarity_{value}"]).argsort()[-n:][::-1] 
    
    values = results[f"similarity_{value}"].iloc[idx_sorted]
    molecules = results[f"smiles_{value}"].iloc[idx_sorted]
    orig = results[f"smiles_original"].iloc[idx_sorted]
    values = list(np.around(np.array(values), 4))
    att = results[f"{value}"].iloc[idx_sorted]

    im = Draw.MolsToGridImage(
            [Chem.MolFromSmiles(smi) for smi in molecules],
            molsPerRow=4,
            subImgSize=(100, 160),
            legends=[f"{v:.3f}" for a, v in zip(att, values)]
        )

    output_path = f"{results_path[:-4]}_{value}_mol.png"
    im.save(output_path)
    im = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(smi) for smi in orig],
        molsPerRow=4,
        subImgSize=(100, 160),
        legends=[f"0" for v in values],
    )

    output_path = f"{results_path[:-4]}_{value}_mol_orig.png"
    im.save(output_path)
    
parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, required=True)
parser.add_argument("--value", type=float, required=True)
parser.add_argument("--n", type=int, default=4)

args = parser.parse_args()

plot_molecules(args.results_path, args.value, args.n)
