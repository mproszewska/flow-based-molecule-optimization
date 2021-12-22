import rdkit
from rdkit.Chem import Crippen


def calculate_logP(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    logP = Crippen.MolLogP(mol)
    return logP
